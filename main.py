"""
Big-O Calculator v4 — 完全自定义输入 · 6+1 策略链
"""
import sys; sys.dont_write_bytecode = True

import re, math
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sympy as sp
import numpy as np
from scipy.optimize import curve_fit

app = FastAPI(title="Big-O Calculator", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ── Models
class TermItem(BaseModel):
    coefficient: str   # raw string: "1", "1/2", "(n-1)/n"
    function: str      # raw string: "n/2", "sqrt(n)", "n-1"

class SolveRequest(BaseModel):
    terms: List[TermItem]
    g_n: str

class DemoStep(BaseModel):
    title: str; description: str; latex: str

class SolveResponse(BaseModel):
    success: bool; latex_formula: str; final_complexity: str
    demo_steps: List[DemoStep]; method: str; note: Optional[str] = None

# ── Symbols
_n = sp.symbols('n', positive=True, integer=True)
_x = sp.symbols('x', positive=True, real=True)
_p = sp.symbols('p', real=True)
_r = sp.symbols('r')
_m = sp.symbols('m', positive=True)

# ═══════════════ PARSING ═══════════════

def _sanitize(s: str) -> str:
    s = s.strip()
    s = s.replace('（', '(').replace('）', ')')
    # Normalize spacing: "sqrt n" → "sqrt(n)", "log (n)" → "log(n)"
    s = re.sub(r'(sqrt|log|sin|cos|tan|exp|abs)\s+\(', r'\1(', s)
    s = re.sub(r'(sqrt|log|sin|cos|tan|exp|abs)\s+([a-zA-Z0-9]+)', r'\1(\2)', s)
    s = re.sub(r'(?<=[a-zA-Z0-9])\s+(?=\))', '', s)  # "sqrt(n )" → "sqrt(n)"
    # Auto-insert multiplication
    s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)
    s = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', s)
    s = re.sub(r'\)([a-zA-Z])', r')*\1', s)
    s = re.sub(r'(?<![a-zA-Z])([a-zA-Z])(?![a-zA-Z])\(', r'\1*(', s)
    s = re.sub(r'(\d)\(', r'\1*(', s)
    s = re.sub(r'\)(\d)', r')*\1', s)
    s = re.sub(r'\)\(', r')*(', s)
    return s

def _sympy_expr(s: str) -> sp.Expr:
    s = _sanitize(s)
    if not s: raise ValueError("empty")
    s = s.replace('^', '**')
    s = re.sub(r'\bln\b', 'log', s)
    return sp.sympify(s, locals={'n': _n, 'log': sp.log, 'sqrt': sp.sqrt,
                                  'exp': sp.exp, 'pi': sp.pi, 'E': sp.E})

def _parse_safe(s: str) -> Optional[sp.Expr]:
    try: return _sympy_expr(s)
    except ValueError: return None

def _parse_coeff(s: str) -> float:
    """Parse coefficient string → numeric approximation."""
    s = s.strip()
    try: return float(eval(s, {"__builtins__": {}}))
    except: pass
    expr = _parse_safe(s)
    if expr is None: return 1.0
    if not expr.has(_n):
        return float(expr.evalf())
    try:
        lim = sp.limit(expr, _n, sp.oo)
        if lim.is_number: return float(lim.evalf())
    except: pass
    try:
        f = sp.lambdify(_n, expr, 'numpy')
        v = f(1000.0)
        if np.isfinite(v): return float(v)
    except: pass
    return 1.0

def _display_coeff(s: str) -> str:
    """Pretty-print a coefficient string."""
    s = s.strip()
    try:
        v = float(eval(s, {"__builtins__": {}}))
        return _clean_coeff(v)
    except: pass
    expr = _parse_safe(s)
    if expr is not None and not expr.has(_n):
        return _clean_coeff(float(expr.evalf()))
    if expr is not None:
        return sp.latex(expr)
    return s

# ═══════════════ CLASSIFICATION ═══════════════

def _is_div(f_str: str) -> Optional[float]:
    m = re.match(r'^\s*n\s*/\s*(\d+\.?\d*)\s*$', f_str)
    if m: return float(m.group(1))
    expr = _parse_safe(f_str)
    if expr is None: return None
    ratio = sp.simplify(expr / _n)
    if ratio.is_number and ratio != 0:
        b = 1.0 / float(ratio)
        if b > 1: return b
    return None

def _is_sub(f_str: str) -> Optional[float]:
    m = re.match(r'^\s*n\s*-\s*(\d+\.?\d*)\s*$', f_str)
    if m: return float(m.group(1))
    expr = _parse_safe(f_str)
    if expr is None: return None
    diff = sp.simplify(_n - expr)
    if diff.is_number and diff > 0: return float(diff)
    return None

def _classify(terms: List[TermItem]) -> Tuple[str, list, list]:
    d_c, s_c = 0, 0
    params, ks = [], []
    for t in terms:
        k = _parse_coeff(t.coefficient)
        if abs(k) < 1e-12: continue  # skip zero-coefficient terms
        b, c = _is_div(t.function), _is_sub(t.function)
        if b is not None: params.append(b); ks.append(k); d_c += 1
        elif c is not None: params.append(c); ks.append(k); s_c += 1
        else: return "mixed", [], [_parse_coeff(x.coefficient) for x in terms if abs(_parse_coeff(x.coefficient))>=1e-12]
    if s_c == 0: return "division", params, ks
    if d_c == 0: return "subtraction", params, ks
    return "mixed", params, ks

# ═══════════════ LATEX FORMATTERS ═══════════════

def _build_formula(req: SolveRequest) -> str:
    parts = []
    for t in req.terms:
        c = _display_coeff(t.coefficient)
        f = _sanitize(t.function)
        f = f.replace('**', '^').replace('*', ' \\cdot ')
        f = re.sub(r'\blog\b', r'\\log', f)
        f = re.sub(r'\bsqrt\b', r'\\sqrt', f)
        if c == '1': parts.append(f"T({f})")
        else: parts.append(f"{c} T({f})")
    gs = _sanitize(req.g_n).replace('**', '^').replace('*', ' \\cdot ')
    gs = re.sub(r'\blog\b', r'\\log', gs)
    gs = re.sub(r'\bsqrt\b', r'\\sqrt', gs)
    return "T(n) = " + (" + ".join(parts) if parts else "0") + " + " + gs

def _clean_exp(e: float) -> str:
    if abs(e) < 1e-6: return "0"
    if abs(e - round(e)) < 1e-6: return str(int(round(e)))
    s = f"{e:.4f}".rstrip('0').rstrip('.'); return s

def _symbolic_log(a_num: float, b_num: float) -> str:
    if abs(b_num - 1.0) < 1e-9: return _clean_exp(math.log(a_num) / math.log(b_num))
    a_i = int(a_num) if abs(a_num - int(a_num)) < 1e-9 else None
    b_i = int(b_num) if abs(b_num - int(b_num)) < 1e-9 else None
    if a_i is None or b_i is None or a_i <= 0 or b_i <= 1: return _clean_exp(math.log(a_num) / math.log(b_num))
    for base in range(2, max(a_i, b_i) + 1):
        x = round(math.log(a_i) / math.log(base)); y = round(math.log(b_i) / math.log(base))
        if base**x == a_i and base**y == b_i:
            if x == y: return "1"
            if y == 1: return str(x)
            if x % y == 0: return str(x // y)
            return f"\\frac{{{x}}}{{{y}}}"
    val = math.log(a_num) / math.log(b_num)
    try:
        sym = sp.nsimplify(val, [sp.sqrt(i) for i in [2,3,5,6,7,10]])
        if sym.is_Rational and sym.q <= 16: return _clean_exp(float(sym))
    except: pass
    if a_i == b_i: return "1"
    return f"\\log_{{{b_i}}} {a_i}"

def _clean_coeff(x: float) -> str:
    if abs(x) < 1e-12: return "0"
    if abs(x - 1.0) < 1e-12: return "1"
    if abs(x - round(x)) < 1e-12: return str(int(round(x)))
    try:
        rat = sp.nsimplify(x, [sp.sqrt(2), sp.sqrt(3), sp.sqrt(5)])
        if rat.is_Rational and rat.q <= 100:
            if rat.q == 1: return str(rat.p)
            return f"\\frac{{{rat.p}}}{{{rat.q}}}"
    except: pass
    return f"{x:.6g}"

def _fmt_theta(p_val, log_pow: int = 0, prefix: str = "\\Theta") -> str:
    eps = 1e-6; str_mode = isinstance(p_val, str)
    if not str_mode:
        if abs(p_val) < eps: p_val = 0.0
        elif abs(p_val - round(p_val)) < eps: p_val = float(round(p_val))
        has_n = abs(p_val) > eps
    else: has_n = p_val != "0"
    has_log = log_pow > 0
    if not has_n and not has_log: return f"{prefix}(1)"
    if not str_mode:
        n_part = "" if not has_n else ("n" if abs(p_val-1.0)<eps else
                 f"n^{{{int(p_val)}}}" if p_val==int(p_val) else
                 f"n^{{{f'{p_val:.4f}'.rstrip('0').rstrip('.')}}}")
    else:
        n_part = "" if not has_n else ("n" if p_val=="1" else f"n^{{{p_val}}}")
    log_part = ("\\log n" if log_pow==1 else f"\\log^{{{log_pow}}} n") if has_log else ""
    if n_part and log_part: inner = f"{n_part} \\cdot {log_part}"
    else: inner = n_part or log_part
    return f"{prefix}({inner})"

def _fmt_exp(base: float, prefix: str = "\\Theta") -> str:
    phi = (1 + math.sqrt(5)) / 2
    if abs(base - phi) < 1e-6:
        return f"{prefix}\\left(\\frac{{1+\\sqrt{{5}}}}{{2}}\\right)^n"
    b = _clean_exp(base)
    return f"{prefix}(1)" if b == "1" else f"{prefix}({b}^n)"

def _big_o_from_expr(expr: sp.Expr) -> str:
    simp = sp.simplify(expr)
    for s in list(simp.free_symbols):
        if s != _n: simp = simp.subs(s, 1)
    simp = sp.simplify(simp)
    # Candidates including resonance forms n·c^n
    # Candidates with numeric ratio check (small-n to avoid overflow)
    for label, cand, n_small, n_large in [
        ("O(n \\cdot 3^n)",_n*3**_n, 10, 30),
        ("O(n \\cdot 2^n)",_n*2**_n, 10, 30),
        ("O(3^n)",3**_n, 10, 30),
        ("O(2^n)",2**_n, 10, 30),
        ("O(n!)",sp.factorial(_n), 5, 10),
        ("O(n^3)",_n**3, 1e3, 1e5),
        ("O(n^2)",_n**2, 1e3, 1e5),
        ("O(n \\log^2 n)",_n*sp.log(_n)**2, 1e3, 1e5),
        ("O(n \\log n)",_n*sp.log(_n), 1e3, 1e5),
        ("O(n)",_n, 1e3, 1e5),
        ("O(\\log^2 n)",sp.log(_n)**2, 1e3, 1e5),
        ("O(\\log n)",sp.log(_n), 1e3, 1e5),
        ("O(\\log \\log n)",sp.log(sp.log(_n)), 1e3, 1e5),
        ("O(1)",1, 1e3, 1e5),
    ]:
        try:
            _fc = sp.lambdify(_n, simp/cand, 'numpy')
            v1, v2 = _fc(n_small), _fc(n_large)
            if np.isfinite(v1) and np.isfinite(v2):
                r = abs(v2) / max(abs(v1), 1e-12)
                # O(1) needs stricter ratio (nearly constant)
                if label == "O(1)" and r > 1.05:
                    continue
                if 0.75 < r < 1.35 and abs(v2) > 0.001:
                    return label
        except: pass
    # Polynomial detection via numeric log-log slope
    try:
        _fp = sp.lambdify(_n, simp, 'numpy')
        vp1, vp2 = abs(_fp(1e6)), abs(_fp(1e8))
        if np.isfinite(vp2) and vp2 > 1e-6 and vp1 > 1e-6:
            pv = np.log(vp2/vp1) / np.log(1e8/1e6)  # slope in log-log
            if pv > 0.05:
                log_k = 0
                _b0 = sp.lambdify(_n, simp/_n**pv, 'numpy')
                u01, u02 = abs(_b0(1e6)), abs(_b0(1e8))
                if np.isfinite(u02) and u02 > 1.01 * max(u01, 1e-9):
                    for k in range(1, 4):
                        try:
                            _fk = sp.lambdify(_n, simp/(_n**pv*sp.log(_n)**k), 'numpy')
                            w1, w2 = abs(_fk(1e6)), abs(_fk(1e8))
                            r = w2/max(w1,1e-12) if w1>1e-12 else 99
                            if np.isfinite(w2) and 0.5<r<2.0 and 0.001<w2<1000: log_k=k; break
                            elif np.isfinite(w2) and w2<0.01: log_k=k-1; break
                        except: continue
                elif np.isfinite(u02) and 0.5 < u02/max(u01,1e-9) < 2.0: log_k=0
                return _fmt_theta(pv, log_k, prefix="O")
    except: pass
    # Exponential detection via numeric growth rate
    try:
        _fe = sp.lambdify(_n, simp, 'numpy')
        ve1, ve2 = abs(_fe(500)), abs(_fe(1000))
        if np.isfinite(ve2) and ve2 > 1e-6 and ve1 > 1e-6:
            gr = np.log(ve2/ve1) / 500  # growth rate per n
            be = np.exp(gr)
            if be > 1.05:
                # Resonance check
                _fr = sp.lambdify(_n, simp/(_n*be**_n), 'numpy')
                vr1, vr2 = abs(_fr(500)), abs(_fr(1000))
                rr = vr2/max(vr1,1e-12) if vr1>1e-12 else 99
                if np.isfinite(vr2) and 0.5<rr<2.0 and 0.001<vr2<1000:
                    return f"O(n \\cdot {_clean_exp(be)}^n)"
                return _fmt_exp(be, prefix="O")
    except: pass
    return f"O\\left({sp.latex(simp)}\\right)"

def _back_substitute(big_o: str) -> str:
    """Convert S(m)→T(n)=S(log_2 n).  m→log n:  n^p→(log n)^p, log n→log log n, c^n→n^{log c}."""
    V = '\uE000V\uE000'
    # Handle c^n → n^{log_2 c}: extract base, replace c^n → n^{log_2 c}
    result = re.sub(r'(?<!\\)\((\d+\.?\d*)\^n\)', lambda m: f'n^{{{_clean_exp(math.log(float(m.group(1)))/math.log(2))}}}', big_o)
    result = re.sub(r'(\d+\.?\d*)\^n', lambda m: f'n^{{{_clean_exp(math.log(float(m.group(1)))/math.log(2))}}}', result)
    # Step 0: n^{...} → (V)^{...}
    result = re.sub(r'(?<!\\)n\^\{', '('+V+')^{', result)
    # Step 1: \log n → \log V
    result = result.replace('\\log n', '\\log '+V)
    # Step 2: standalone n → V
    result = re.sub(r'(?<!\\)(?<!\w)n(?!\w)', V, result)
    # Step 3: V → \log n
    result = result.replace(V, '\\log n')
    # Step 4: \cdot between consecutive \log
    result = re.sub(r'(\\log\s+n)\s+(\\log)', r'\1 \\cdot \2', result)
    return result

# ═══════════════ STRATEGY 0: MASTER THEOREM ═══════════════

def _master_theorem(a: float, b: float, g_sym: sp.Expr) -> Optional[Tuple[str, List[DemoStep], Optional[str]]]:
    steps = []; alpha = math.log(a)/math.log(b); alpha_s = _symbolic_log(a,b)
    steps.append(DemoStep(title="主定理 — 参数提取",
        description=f"递推 $T(n)={a:.6g}\\,T(n/{b:.6g})+g(n)$。",
        latex=f"a={a:.6g},\\;b={b:.6g},\\;g(n)={sp.latex(g_sym)}"))
    steps.append(DemoStep(title="主定理 — 临界指数",
        description="$\\log_b a$ 判定阈值。", latex=f"\\log_b a={alpha_s}"))
    try: g_beta=float(sp.limit(sp.log(g_sym)/sp.log(_n),_n,sp.oo))
    except: g_beta=None
    case,log_k=None,0
    if g_beta is not None and g_beta<alpha-1e-9: case=1
    elif g_beta is not None and abs(g_beta-alpha)<1e-9:
        case=3
        # Fast numeric log-k: check ratio stability at two n values
        try:
            _base = sp.lambdify(_n, g_sym/_n**alpha, 'numpy')
            v0_1, v0_2 = abs(_base(1e6)), abs(_base(1e8))
            if np.isfinite(v0_2) and v0_2 > 1.01 * max(v0_1, 1e-9):
                # v0 grows → k=0 doesn't match, test higher k
                for k in range(1, 5):
                    _fk = sp.lambdify(_n, g_sym/(_n**alpha*sp.log(_n)**k), 'numpy')
                    v1, v2 = abs(_fk(1e6)), abs(_fk(1e8))
                    ratio = v2 / max(v1, 1e-12) if v1 > 1e-12 else 99
                    if np.isfinite(v2) and 0.5 < ratio < 2.0 and 0.001 < v2 < 1000:
                        log_k, case = k, 2; break
                    elif np.isfinite(v2) and v2 < 0.01:
                        log_k, case = k-1, 2; break
            elif np.isfinite(v0_2) and 0.5 < v0_2/max(v0_1, 1e-9) < 2.0:
                log_k, case = 0, 2
        except: pass
    elif g_beta is not None: case=3
    if case is None: return None
    if case==1:
        steps.append(DemoStep(title="主定理 — 情况一",
            description=f"$g(n)=O(n^{{{alpha_s}-\\varepsilon}})$。", latex=f"\\lim g/n^{{{alpha_s}}}=0"))
        big_o=_fmt_theta(alpha_s)
        steps.append(DemoStep(title="主定理 — 结论", description="齐次主导。", latex=f"T(n)={big_o}"))
        return big_o,steps,None
    elif case==2:
        steps.append(DemoStep(title="主定理 — 情况二",
            description=f"$g(n)=\\Theta(n^{{{alpha_s}}}\\log^{{{log_k}}} n)$。",
            latex=f"\\lim g/(n^{{{alpha_s}}}\\log^{{{log_k}}} n)=c\\neq0"))
        big_o=_fmt_theta(alpha_s,log_k+1)
        steps.append(DemoStep(title="主定理 — 结论", description="增加对数因子。", latex=f"T(n)={big_o}"))
        return big_o,steps,None
    else:
        steps.append(DemoStep(title="主定理 — 情况三",
            description=f"$g(n)=\\Omega(n^{{{alpha_s}+\\varepsilon}})$。", latex=f"\\lim g/n^{{{alpha_s}}}=\\infty"))
        return None

# ═══════════════ STRATEGY 1: AKRA-BAZZI ═══════════════

def _akra_bazzi(b_vals: List[float], k_vals: List[float], g_sym: sp.Expr) -> Optional[Tuple[str, List[DemoStep], Optional[str]]]:
    steps = []
    term_disp = " + ".join(f"{_clean_coeff(kv)}\\,T(n/{_clean_coeff(bv)})" for kv,bv in zip(k_vals,b_vals))
    steps.append(DemoStep(title="Akra-Bazzi — 方程识别",
        description="除法缩减型 $n/b_i$。", latex=f"T(n)={term_disp}+{sp.latex(g_sym)}"))
    eq = sum(k*(1/b)**_p for k,b in zip(k_vals,b_vals))-1
    p_parts = []
    for k,b in zip(k_vals,b_vals):
        ks=_clean_coeff(k); bf=_clean_coeff(1.0/b)
        p_parts.append(f"{ks}\\cdot\\left({bf}\\right)^p" if ks!="1" else f"\\left({bf}\\right)^p")
    steps.append(DemoStep(title="Akra-Bazzi — p 方程",
        description="$\\sum k_i b_i^{-p}=1$。", latex=" + ".join(p_parts)+" = 1"))

    p_val,p_display,p_is_exact=None,None,False
    try:
        for sol in sp.solve(eq,_p):
            if sol.is_real:
                ps=sp.simplify(sol); p_val=float(ps.evalf()); p_is_exact=True
                p_display=_clean_exp(p_val) if ps.is_Number else sp.latex(ps); break
    except: pass
    if p_val is None:
        for guess in [0.0,0.5,1.0,1.5,2.0,-0.5,0.25,3.0]:
            try: p_val=float(sp.nsolve(eq,guess,tol=1e-14,maxsteps=200)); break
            except: continue
    if p_val is None: return None
    if not p_is_exact:
        try: p_high=float(sp.nsolve(eq,p_val,tol=1e-30,maxsteps=500,prec=60))
        except: p_high=p_val
    else: p_high=p_val
    pd_demo = f"{p_high:.16f}" if not p_is_exact else p_display
    steps.append(DemoStep(title="Akra-Bazzi — 求解 p",
        description=("符号精确解。" if p_is_exact else "超越方程高精度数值解。"),
        latex=f"p={pd_demo}"))

    integrand = g_sym.subs(_n,_x)/_x**(p_val+1)
    steps.append(DemoStep(title="Akra-Bazzi — 构造积分",
        description="$I(n)=\\int_1^n g(x)/x^{p+1}dx$。",
        latex=f"I(n)=\\int_1^n\\frac{{{sp.latex(g_sym)}}}{{x^{{{_clean_exp(p_val+1)}}}}}\\,dx"))
    integral_expr=None
    if p_is_exact:
        try:
            integral_expr=sp.simplify(sp.integrate(integrand,(_x,1,_n)))
            ishow=integral_expr
            for atom in ishow.atoms(sp.Float): ishow=ishow.subs(atom,sp.Float(round(float(atom),4)))
            steps.append(DemoStep(title="Akra-Bazzi — 积分闭式解",
                description="SymPy 求得解析式。", latex=f"I(n)={sp.latex(ishow)}"))
        except Exception:
            try:
                g_b=float(sp.limit(sp.log(g_sym)/sp.log(_n),_n,sp.oo))
                steps.append(DemoStep(title="Akra-Bazzi — 渐近积分分析",
                    description=f"g(n) 主导指数 α≈{_clean_exp(g_b)}。",
                    latex=f"\\text{{g(n) 主导项: }}n^{{{_clean_exp(g_b)}}}"))
                if g_b<p_val: integral_expr=sp.Integer(0)
            except: return None
    else:
        steps.append(DemoStep(title="Akra-Bazzi — 跳过积分",
            description="p 为超越数，跳过符号积分，直接 case 分析。", latex=""))

    # ── Final: case analysis ──
    try:
        _fg = sp.lambdify(_n, g_sym, 'numpy')
        vg1, vg2 = abs(_fg(1e6)), abs(_fg(1e8))
        g_beta = np.log(vg2/vg1) / np.log(1e8/1e6) if vg1>1e-6 and vg2>1e-6 else 0.0
    except: g_beta = 0.0

    big_o = None
    if integral_expr is not None and p_is_exact:
        has_floats = any(isinstance(a, sp.Float) for a in integral_expr.atoms()) if hasattr(integral_expr, 'atoms') else False
        if not has_floats:
            final = _n**p_val * (1 + sp.simplify(integral_expr))
            big_o = _big_o_from_expr(final)

    if big_o is None:
        if g_beta < p_val - 1e-6:
            big_o = _fmt_theta(p_val)
        elif abs(g_beta - p_val) < 1e-6:
            if p_is_exact:
                log_k = 0
                try:
                    _g0 = sp.lambdify(_n, g_sym / _n**p_val, 'numpy')
                    v01, v02 = abs(_g0(1e6)), abs(_g0(1e8))
                    if np.isfinite(v02) and v02 > 1.01 * max(v01, 1e-9):
                        for k in range(1, 4):
                            _gk = sp.lambdify(_n, g_sym / (_n**p_val * sp.log(_n)**k), 'numpy')
                            v1, v2 = abs(_gk(1e6)), abs(_gk(1e8))
                            r = v2 / max(v1, 1e-12) if v1 > 1e-12 else 99
                            if np.isfinite(v2) and 0.5 < r < 2.0 and 0.001 < v2 < 1000:
                                log_k = k; break
                            elif np.isfinite(v2) and v2 < 0.01:
                                log_k = k - 1; break
                    elif np.isfinite(v02) and 0.5 < v02 / max(v01, 1e-9) < 2.0:
                        log_k = 0
                except: pass
                big_o = _fmt_theta(p_val, log_k + 1)
            else:
                big_o = _fmt_theta(p_val, 1)
        else:
            big_o = _big_o_from_expr(g_sym)

    if p_is_exact: note=None
    else:
        en_parts=[]
        for k,b in zip(k_vals,b_vals):
            ks=_clean_coeff(k)
            en_parts.append(f"({_clean_coeff(1.0/b)})^p" if ks=="1" else f"{ks}({_clean_coeff(1.0/b)})^p")
        note=f"\\text{{其中 }} p \\approx {p_high:.16f} \\quad\\left(\\text{{满足 }} {' + '.join(en_parts)} = 1\\right)"
    steps.append(DemoStep(title="Akra-Bazzi — 组合结论",
        description="$\\Theta(n^p(1+I(n)))$ 经 case 分析得最终复杂度。", latex=f"T(n)={big_o}"))
    return big_o,steps,note

# ═══════════════ STRATEGY 2: LINEAR RECURRENCE ═══════════════

def _linear_recurrence(c_vals: List[float], k_vals: List[float], g_sym: sp.Expr) -> Optional[Tuple[str, List[DemoStep], Optional[str]]]:
    steps = []; max_c=int(max(c_vals)); c_ints=[int(c) for c in c_vals]
    T=sp.Function('T'); rhs=sum(k*T(_n-ci) for k,ci in zip(k_vals,c_ints))+g_sym
    td=" + ".join(f"{_clean_coeff(kv)}\\,T(n-{cv})" for kv,cv in zip(k_vals,c_ints))
    steps.append(DemoStep(title="线性递推 — 方程识别",
        description="减法缩减型 $n-c_i$。", latex=f"T(n)={td}+{sp.latex(g_sym)}"))
    try:
        sol=sp.rsolve(T(_n)-rhs,T(_n))
        if sol is not None:
            ss=sp.simplify(sol)
            steps.append(DemoStep(title="线性递推 — rsolve 闭式解",
                description="SymPy 直接求得闭式解。", latex=f"T(n)={sp.latex(ss)}"))
            big_o=_big_o_from_expr(ss)
            steps.append(DemoStep(title="线性递推 — 渐近阶", description="", latex=f"T(n)={big_o}"))
            return big_o,steps,None
    except: pass
    char_eq=_r**max_c-sum(k*_r**(max_c-ci) for k,ci in zip(k_vals,c_ints))
    steps.append(DemoStep(title="线性递推 — 特征方程", description="",
        latex=sp.latex(sp.simplify(char_eq))+" = 0"))
    try: roots=sp.nroots(char_eq)
    except:
        try: roots=[sp.N(s) for s in sp.solve(char_eq,_r)]
        except: return None
    rmods=[abs(float(rt)) for rt in roots]; max_mod=max(rmods)
    # Multiplicity: count roots with the same VALUE (not just same modulus)
    root_vals = [complex(rt.evalf()) for rt in roots]
    mult = 0
    for rv in root_vals:
        # Count roots equal to the one with max real part among those with max modulus
        pass
    # Simpler: compare actual root values within tolerance
    mult = max(sum(1 for rv2 in root_vals if abs(rv - rv2) < 1e-6) for rv in root_vals)
    steps.append(DemoStep(title="线性递推 — 特征根",
        description=f"{len(roots)} 根, 最大模 {max_mod:.6g} (重数{mult})。",
        latex=",\\ ".join(f"r_{{{i+1}}}={float(rt):.6g}" for i,rt in enumerate(roots))))
    if max_mod>1.001:
        if mult>1:
            b_str = _clean_exp(max_mod)
            homo = f"\\Theta(n \\cdot {b_str}^n)" if abs(max_mod - round(max_mod)) < 1e-6 else f"\\Theta(n \\cdot ({b_str})^n)"
        else:
            homo = _fmt_exp(max_mod)
    elif abs(max_mod-1)<1e-3: homo=_fmt_theta(mult-1) if mult>1 else "\\Theta(1)"
    else: homo="\\Theta(1)"
    steps.append(DemoStep(title="线性递推 — 齐次解", description="", latex=f"T_h(n)={homo}"))
    if g_sym!=0:
        try:
            g_b=float(sp.limit(sp.log(g_sym)/sp.log(_n),_n,sp.oo))
            steps.append(DemoStep(title="线性递推 — 非齐次项",
                description=f"g(n) 指数 α≈{_clean_exp(g_b)}。", latex=f"g(n)\\sim n^{{{_clean_exp(g_b)}}}"))
            # Resonance check: g(n) ≈ base^n and base ≈ max_mod
            if max_mod > 1:
                try:
                    g_exp_rate = sp.limit(sp.log(g_sym)/_n, _n, sp.oo)
                    if g_exp_rate.is_finite:
                        g_base = float(sp.exp(g_exp_rate).evalf())
                        if abs(g_base - max_mod) < 0.05:
                            b_str = _clean_exp(max_mod)
                            big_o = f"\\Theta(n \\cdot {b_str}^n)" if abs(max_mod-round(max_mod))<1e-6 else f"\\Theta(n \\cdot ({b_str})^n)"
                            steps.append(DemoStep(title="线性递推 — 共振检测", description="",
                                latex=f"\\text{{特征根底数 }} {max_mod:.6g} \\text{{ 与 }} g(n) \\text{{ 共振}}\\Rightarrow {big_o}"))
                            return big_o, steps, None
                except: pass
                big_o = homo
            elif g_b>0: big_o=_fmt_theta(g_b)
            else: big_o=homo
        except: big_o=homo
    else: big_o=homo
    steps.append(DemoStep(title="线性递推 — 最终结论", description="", latex=f"T(n)={big_o}"))
    return big_o,steps,None

# ═══════════════ STRATEGY 3: DOMAIN SUBSTITUTION ═══════════════

def _domain_substitution(terms: List[TermItem], g_sym: sp.Expr) -> Optional[Tuple[str, List[DemoStep], Optional[str]]]:
    steps = []
    has_nl = any('sqrt' in t.function or '**' in t.function.replace('^','**') for t in terms)
    if not has_nl: return None
    steps.append(DemoStep(title="换元构造法 — 识别非线性",
        description="检测到 $\\sqrt{n}$ 等，令 $n=2^m$。",
        latex="n=2^m,\\;S(m)=T(2^m)"))
    new_terms = []
    for t in terms:
        fs = _sanitize(t.function.strip()).replace('^','**')
        fs = re.sub(r'sqrt\s*\(\s*n\s*\)','2**(m/2)',fs)
        # Also catch "sqrt n" (spaced, no parens — already sanitized to sqrt(n))
        fs = re.sub(r'n\s*\*\*\s*0\.5','2**(m/2)',fs)
        new_terms.append(TermItem(coefficient=t.coefficient, function=fs))
    simplified = False
    for t in new_terms:
        mt = re.search(r'2\s*\*\*\s*\(\s*m\s*/\s*(\d+)\s*\)', t.function)
        if mt:
            t.function = f"n/{int(mt.group(1))}"; simplified = True
    if not simplified: return None
    g_m = sp.simplify(g_sym.subs(_n,2**_m).subs(_m,_n))
    steps.append(DemoStep(title="换元构造法 — 代换完成",
        description="$n=2^m$ 代入得标准递推。",
        latex=f"S(n)={' + '.join(f'{_display_coeff(t.coefficient)} S({t.function})' for t in new_terms)}+{sp.latex(g_m)}"))
    strat,params,ks = _classify(new_terms)
    if strat=="division":
        r=_akra_bazzi(params,ks,g_m)
        if r:
            big_o,ab_steps,note=r; steps.extend(ab_steps)
            big_o=_back_substitute(big_o)
            if note: note=_back_substitute(note)
            steps.append(DemoStep(title="换元构造法 — 回代",
                description="$m=\\log_2 n$, $T(n)=S(\\log_2 n)$。", latex=f"T(n)={big_o}"))
            return big_o,steps,note
    elif strat=="subtraction":
        r=_linear_recurrence(params,ks,g_m)
        if r:
            big_o,ln_steps,note=r; steps.extend(ln_steps)
            big_o=_back_substitute(big_o)
            if note: note=_back_substitute(note)
            steps.append(DemoStep(title="换元构造法 — 回代",
                description="$m=\\log_2 n$, $T(n)=S(\\log_2 n)$。", latex=f"T(n)={big_o}"))
            return big_o,steps,note
    return None

# ═══════════════ STRATEGY 4: CONTINUOUS APPROX ═══════════════

def _continuous_approx(terms: List[TermItem], g_sym: sp.Expr) -> Optional[Tuple[str, List[DemoStep], Optional[str]]]:
    if len(terms)!=1: return None
    t=terms[0]; c=_is_sub(t.function)
    if c is None: return None
    k=_parse_coeff(t.coefficient)
    if abs(k-1.0)>1e-9: return None
    steps=[DemoStep(title="连续化微积分 — 启动",
        description=f"$T(n)-T(n-{int(c)})\\approx g(n)$。",
        latex=f"T(n)-T(n-{int(c)})\\approx {sp.latex(g_sym)}")]
    T_func=sp.Function('T')
    de=sp.Eq(sp.Derivative(T_func(_n),_n), g_sym/c)
    steps.append(DemoStep(title="连续化微积分 — 微分方程",
        description=f"$dT/dn\\approx g(n)/{int(c)}$。", latex=sp.latex(de)))
    try:
        sol=sp.dsolve(de,T_func(_n))
        if sol is not None:
            steps.append(DemoStep(title="连续化微积分 — dsolve",
            description="SymPy 求得通解。", latex=sp.latex(sol)))
        big_o=_big_o_from_expr(sp.simplify(sol.rhs))
        steps.append(DemoStep(title="连续化微积分 — 渐近阶",
            description="提取主导项。", latex=f"T(n)\\approx{big_o}"))
        return big_o,steps,None
    except Exception:
        steps.append(DemoStep(title="连续化微积分 — 失败", description="", latex=""))
        return None

# ═══════════════ STRATEGY 5: POINCARÉ-PERRON ═══════════════

def _poincare_perron(terms: List[TermItem], g_sym: sp.Expr) -> Optional[Tuple[str, List[DemoStep], Optional[str]]]:
    has_var = False
    for t in terms:
        expr=_parse_safe(t.function)
        if expr is not None and expr.has(_n) and expr!=_n:
            if _is_div(t.function) is None and _is_sub(t.function) is None:
                has_var=True; break
    if not has_var: return None
    steps=[DemoStep(title="庞加莱-佩隆 — 启动",
        description="检测到非常系数变元，$n\\to\\infty$ 极限分析。",
        latex="\\text{分析各项系数的极限行为}")]
    lim_terms=[]
    for t in terms:
        expr=_parse_safe(t.function)
        if expr is None: continue
        try:
            ratio=sp.limit(expr/_n,_n,sp.oo)
            if ratio.is_number:
                k=_parse_coeff(t.coefficient)
                lim_terms.append((k,float(ratio)))
                steps.append(DemoStep(title="庞加莱-佩隆 — 极限分析",
                    description=f"$\\lim f_i(n)/n={_clean_exp(float(ratio))}$。",
                    latex=f"\\lim\\frac{{{sp.latex(expr)}}}{{n}}={_clean_exp(float(ratio))}"))
        except: continue
    if not lim_terms: return None
    all_div=all(0<r<1 for _,r in lim_terms)
    all_sub=all(abs(r-1)<1e-6 for _,r in lim_terms)
    if all_div:
        b_lim=[1.0/r for _,r in lim_terms]; k_lim=[k for k,_ in lim_terms]
        r=_akra_bazzi(b_lim,k_lim,g_sym)
        if r:
            big_o,ab_steps,note=r; steps.extend(ab_steps)
            return big_o,steps,note
    elif all_sub:
        c_lim=[1 for _ in lim_terms]; k_lim=[k for k,_ in lim_terms]
        r=_linear_recurrence(c_lim,k_lim,g_sym)
        if r:
            big_o,ln_steps,note=r; steps.extend(ln_steps)
            return big_o,steps,note
    return None

# ═══════════════ PREPROCESSING ═══════════════

def _preprocess_terms(terms: List[TermItem], g_sym: sp.Expr) -> Tuple[List[TermItem], sp.Expr, str]:
    tn_coeff=0.0; other=[]
    for t in terms:
        fs=t.function.strip()
        if fs=='n' or re.match(r'^\s*n\s*\*\*\s*1\s*$',fs) or re.match(r'^\s*n\s*\^\s*1\s*$',fs):
            tn_coeff+=_parse_coeff(t.coefficient)
        else: other.append(t)
    if abs(tn_coeff)<1e-12: return terms,g_sym,""
    left=1.0-tn_coeff
    if abs(left)<1e-12: return [],g_sym,"方程系数冲突退化: 两侧 T(n) 抵消，无法构成有效递推关系"
    norm_terms=[TermItem(coefficient=str(_parse_coeff(t.coefficient)/left), function=t.function) for t in other]
    norm_g=sp.simplify(g_sym/left)
    return norm_terms,norm_g,""

# ═══════════════ NUMERICAL FALLBACK ═══════════════

FIT_MODELS = [
    ("O(1)",        lambda n,a: np.full_like(n,a),                    [1.0]),
    ("O(\\log n)",  lambda n,a,b: a*np.log(np.maximum(n,1e-9))+b,     [1.0,0.0]),
    ("O(n)",        lambda n,a,b: a*n+b,                               [1.0,0.0]),
    ("O(n \\log n)",lambda n,a,b: a*n*np.log(np.maximum(n,1e-9))+b,   [1.0,0.0]),
    ("O(n^2)",      lambda n,a,b,c: a*n**2+b*n+c,                      [1.0,0.1,0.0]),
    ("O(n^3)",      lambda n,a,b,c: a*n**3+b*n**2+c,                   [0.01,0.0,0.0]),
    ("O(2^n)",      lambda n,a,b: a*2.0**n+b,                          [1e-6,0.0]),
    ("O(3^n)",      lambda n,a,b: a*3.0**n+b,                          [1e-12,0.0]),
]

def _eval_fn_str(f_str: str, n_val: float) -> float:
    s=f_str.strip()
    m=re.match(r'^\s*n\s*/\s*(\d+\.?\d*)\s*$',s)
    if m: return math.floor(n_val/float(m.group(1)))
    m=re.match(r'^\s*n\s*-\s*(\d+\.?\d*)\s*$',s)
    if m: return max(0.0,n_val-float(m.group(1)))
    s=s.replace('^','**')
    safe={"n":n_val,"log":math.log,"sqrt":math.sqrt,"exp":math.exp,"pi":math.pi}
    try:
        val=float(eval(s,{"__builtins__":{}},safe))
        return math.floor(val) if val!=int(val) else val
    except: return n_val

def _compute_seq(terms: List[TermItem], g_func, N: int=1000) -> np.ndarray:
    T=np.zeros(N+1,dtype=np.float64); T[0],T[1]=0.0,1.0
    for i in range(2,N+1):
        val=0.0
        for t in terms:
            fi=_eval_fn_str(t.function,float(i))
            idx=int(max(0,min(i-1,math.floor(fi))))
            val+=_parse_coeff(t.coefficient)*T[idx]
        val+=g_func(float(i))
        if not np.isfinite(val) or abs(val)>1e300: val=1e300
        T[i]=val
    return T

def _numerical_solve(terms: List[TermItem], g_sym: sp.Expr) -> Optional[Tuple[str, List[DemoStep], Optional[str]]]:
    steps=[DemoStep(title="数值模拟 — 启动",
        description="6 策略均未解析，启用数值拟合。", latex="\\text{切换至数值求解器…}")]
    N=1000
    g_func=sp.lambdify(_n,g_sym,modules=["numpy",{"log":np.log,"sqrt":np.sqrt}])
    T=_compute_seq(terms,g_func,N)
    nv=np.arange(1,N+1,dtype=np.float64); tv=T[1:].astype(np.float64)
    mask=np.isfinite(tv)&(tv<1e290); nv,tv=nv[mask],tv[mask]
    if len(tv)<5:
        steps.append(DemoStep(title="数值模拟 — 序列退化",description="有效点不足。",latex=""))
        return "\\Theta(1)",steps,"序列退化，降级输出"
    tv_range=float(np.max(tv)-np.min(tv))
    if tv_range<1e-12:
        steps.append(DemoStep(title="数值模拟 — 序列恒常",
            description=f"全为常数。", latex=f"T(n)\\equiv{tv[0]:.6g}"))
        return "\\Theta(1)",steps,"序列恒常，无需拟合"
    steps.append(DemoStep(title="数值模拟 — 序列生成",
        description=f"T(1)…T({len(nv)})。", latex=f"T({len(nv)})={tv[-1]:.6e}"))

    best_name,best_r2="\\Theta(1)",-np.inf; results_table=[]; all_failed=True
    for name,func,p0 in FIT_MODELS:
        try:
            popt,_=curve_fit(func,nv,tv,p0=p0,maxfev=20000)
            pred=func(nv,*popt)
            if not np.all(np.isfinite(pred)): continue
            ss_res=np.sum((tv-pred)**2); ss_tot=np.sum((tv-np.mean(tv))**2)
            r2=1-ss_res/ss_tot if ss_tot>1e-300 else 0
            if np.isfinite(r2):
                results_table.append((name,r2)); all_failed=False
                if r2>best_r2: best_r2,best_name=r2,name
        except: continue
    if all_failed:
        steps.append(DemoStep(title="数值模拟 — 拟合失败",description="全模型发散。",latex=""))
        return "\\Theta(1)",steps,"数值波动过大，降级输出"
    rows_latex=" & ".join([n for n,_ in results_table])+" \\\\ "
    rows_latex+=" & ".join([f"{r:.4f}" for _,r in results_table])
    steps.append(DemoStep(title="数值模拟 — 拟合对比",description="",
        latex=f"\\begin{{array}}{{{'c'*len(results_table)}}} {rows_latex} \\end{{array}}"))
    steps.append(DemoStep(title="数值模拟 — 最佳匹配",description="",
        latex=f"\\text{{最佳: }}{best_name},\\;R^2={best_r2:.4f}"))
    steps.append(DemoStep(title="⚠ 提示",description="数值结果建议交叉验证。",latex=""))
    return best_name,steps,"由于符号数学限制，此结果由数值模拟曲线拟合得出"

# ═══════════════ MAIN CHAIN ═══════════════

def _try_all_strategies(req: SolveRequest, g_sym: sp.Expr, strategy: str, params: list, ks: list):
    all_pos = all(k>1e-12 for k in ks) if ks else False

    # Strategy 0: Master Theorem
    if all_pos and strategy in ("division",) and len(params)==1:
        r = _master_theorem(ks[0], params[0], g_sym)
        if r and r[0]:
            big_o, steps, note = r
            return big_o, steps, "主定理", note

    # Strategy 1: Akra-Bazzi
    if all_pos and strategy == "division" and params:
        r = _akra_bazzi(params, ks, g_sym)
        if r and r[0]:
            big_o, steps, note = r
            return big_o, steps, "Akra-Bazzi 定理", note

    # Strategy 2: Linear + Continuous
    if strategy in ("subtraction",):
        if len(params) == 1:
            r = _continuous_approx(req.terms, g_sym)
            if r and r[0]:
                big_o, steps, note = r
                return big_o, steps, "连续化微积分", note
        r = _linear_recurrence(params, ks, g_sym)
        if r and r[0]:
            big_o, steps, note = r
            return big_o, steps, "线性特征方程", note
    else:
        r = _continuous_approx(req.terms, g_sym)
        if r and r[0]:
            big_o, steps, note = r
            return big_o, steps, "连续化微积分", note

    # Strategy 3: Domain Substitution
    r = _domain_substitution(req.terms, g_sym)
    if r and r[0]:
        big_o, steps, note = r
        return big_o, steps, "换元构造法", note

    # Strategy 5: Poincaré-Perron
    r = _poincare_perron(req.terms, g_sym)
    if r and r[0]:
        big_o, steps, note = r
        return big_o, steps, "庞加莱-佩隆", note

    # Strategy 6: Numerical
    r = _numerical_solve(req.terms, g_sym)
    if r and r[0]:
        big_o, steps, note = r
        return big_o, steps, "数值模拟曲线拟合", note

    return None

# ═══════════════ ENDPOINT ═══════════════

@app.post("/api/solve", response_model=SolveResponse)
async def solve(req: SolveRequest):
    if not req.terms: raise HTTPException(400,"至少需要一项递推项")
    latex_formula=_build_formula(req)
    try: g_sym=_sympy_expr(req.g_n)
    except ValueError as e: raise HTTPException(400,str(e))
    cleaned_terms,g_sym,pre_err=_preprocess_terms(req.terms,g_sym)
    if pre_err: raise HTTPException(400,pre_err)
    if all(abs(_parse_coeff(t.coefficient))<1e-12 for t in cleaned_terms):
        raise HTTPException(400,"所有递推系数均为零")
    if cleaned_terms!=req.terms:
        latex_formula=_build_formula(SolveRequest(terms=cleaned_terms,g_n=req.g_n))
    strategy,params,ks=_classify(cleaned_terms)
    result=_try_all_strategies(SolveRequest(terms=cleaned_terms,g_n=req.g_n),g_sym,strategy,params,ks)
    if result is None: raise HTTPException(500,"所有策略均求解失败")
    big_o,steps,method,note=result
    return SolveResponse(success=True,latex_formula=latex_formula,final_complexity=big_o,
                         demo_steps=steps,method=method,note=note)

@app.get("/api/health")
async def health(): return {"status":"ok"}

if __name__=="__main__":
    import uvicorn; uvicorn.run(app,host="0.0.0.0",port=8000)
