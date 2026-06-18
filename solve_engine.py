"""BigOer — 6 symbolic solvers + numerical fallback."""

import re, math
from typing import List, Optional, Tuple

import sympy as sp
import numpy as np
from scipy.optimize import curve_fit

from parse_clean import (_n, _x, _p, _r, _m,
    _sympy_expr, _parse_safe, _parse_coeff, _is_div, _is_sub, _classify,
    _clean_coeff, _display_coeff, _eval_fn_str, _compute_seq)
from format_out import (_clean_exp, _symbolic_log, _fmt_theta, _fmt_exp,
    _detect_log_k, _back_substitute, _big_o_from_expr)
from pydantic import BaseModel  # for DemoStep definition

class DemoStep(BaseModel):
    title: str; description: str; latex: str


# ═══════════════ STRATEGY 0: MASTER THEOREM ═══════════════

def _master_theorem(a: float, b: float, g_sym: sp.Expr) -> Optional[Tuple[str, List[DemoStep], Optional[str]]]:
    steps = []; alpha = math.log(a)/math.log(b); alpha_s = _symbolic_log(a,b)
    steps.append(DemoStep(title="主定理 — 参数提取",
        description=f"递推 $T(n)={a:.6g}\\,T(n/{b:.6g})+g(n)$。",
        latex=f"a={a:.6g},\\;b={b:.6g},\\;g(n)={sp.latex(g_sym)}"))
    steps.append(DemoStep(title="主定理 — 临界指数",
        description="$\\log_b a$ 判定阈值。", latex=f"\\log_b a={alpha_s}"))
    # Use _detect_log_k for robust case detection
    log_k, case = _detect_log_k(g_sym, alpha)
    if case == 0: return None  # ambiguous → fall through to Akra-Bazzi
    if case == 1:
        steps.append(DemoStep(title="主定理 — 情况一",
            description=f"$g(n)=O(n^{{{alpha_s}-\\varepsilon}})$。", latex=f"\\lim g/n^{{{alpha_s}}}=0"))
        big_o = _fmt_theta(alpha_s)
        steps.append(DemoStep(title="主定理 — 结论", description="齐次主导。", latex=f"T(n)={big_o}"))
        return big_o, steps, None
    elif case == 2:
        steps.append(DemoStep(title="主定理 — 情况二",
            description=f"$g(n)=\\Theta(n^{{{alpha_s}}}\\log^{{{log_k}}} n)$。",
            latex=f"\\lim g/(n^{{{alpha_s}}}\\log^{{{log_k}}} n)=c\\neq0"))
        big_o = _fmt_theta(alpha_s, log_k + 1)
        steps.append(DemoStep(title="主定理 — 结论", description="增加对数因子。", latex=f"T(n)={big_o}"))
        return big_o, steps, None
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

    # Final: case analysis via _detect_log_k
    log_k, case = _detect_log_k(g_sym, p_val)

    big_o = None
    if integral_expr is not None and p_is_exact:
        has_floats = any(isinstance(a, sp.Float) for a in integral_expr.atoms()) if hasattr(integral_expr,'atoms') else False
        if not has_floats:
            # Detect log(log(n)) or sqrt(log(n)) in integral → direct Θ(n^p · factor)
            ill_str = str(integral_expr)
            if 'log(log(' in ill_str:
                big_o = f"\\Theta(n^{{{_clean_exp(p_val)}}} \\log \\log n)"
            elif 'sqrt(log(' in ill_str or 'log(n)**(1/2)' in ill_str.replace(' ',''):
                big_o = f"\\Theta(n^{{{_clean_exp(p_val)}}} \\sqrt{{\\log n}})"
            elif 'log(' in ill_str and '**2' in ill_str.replace(' ',''):
                big_o = f"\\Theta(n^{{{_clean_exp(p_val)}}} \\log^{{2}} n)"
            elif 'log(' in ill_str and '**' not in ill_str:
                big_o = f"\\Theta(n^{{{_clean_exp(p_val)}}} \\log n)"
            else:
                final = _n**p_val*(1+sp.simplify(integral_expr)); big_o = _big_o_from_expr(final)

    if big_o is None:
        if case == 1: big_o = _fmt_theta(p_val)
        elif case == 2:
            if p_is_exact: big_o = _fmt_theta(p_val, log_k + 1)
            else: big_o = _fmt_theta(p_val, 1)
        elif case == 3: big_o = _big_o_from_expr(g_sym)
        else: big_o = _fmt_theta(p_val)  # ambiguous (case 0): default to n^p

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

    # rsolve guard: skip for high-order to prevent hangs
    if max_c <= 5:
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
    root_vals=[complex(rt.evalf()) for rt in roots]
    mult=max(sum(1 for rv2 in root_vals if abs(rv-rv2)<1e-6) for rv in root_vals)
    steps.append(DemoStep(title="线性递推 — 特征根",
        description=f"{len(roots)} 根, 最大模 {max_mod:.6g} (重数{mult})。",
        latex=",\\ ".join(f"r_{{{i+1}}}={float(rt):.6g}" for i,rt in enumerate(roots))))
    if max_mod>1.001:
        if mult>1:
            b_str=_clean_exp(max_mod)
            homo=f"\\Theta(n \\cdot {b_str}^n)" if abs(max_mod-round(max_mod))<1e-6 else f"\\Theta(n \\cdot ({b_str})^n)"
        else: homo=_fmt_exp(max_mod)
    elif abs(max_mod-1)<1e-3: homo=_fmt_theta(mult-1) if mult>1 else "\\Theta(1)"
    else: homo="\\Theta(1)"
    steps.append(DemoStep(title="线性递推 — 齐次解", description="", latex=f"T_h(n)={homo}"))
    if g_sym!=0:
        try:
            _fg=sp.lambdify(_n,g_sym,'numpy')
            vg1,vg2=abs(_fg(1e6)),abs(_fg(1e8))
            g_b=np.log(vg2/vg1)/np.log(1e8/1e6) if vg1>1e-6 and vg2>1e-6 else 0.0
            steps.append(DemoStep(title="线性递推 — 非齐次项",
                description=f"g(n) 指数 α≈{_clean_exp(g_b)}。", latex=f"g(n)\\sim n^{{{_clean_exp(g_b)}}}"))
            if max_mod>1:
                try:
                    g_exp_rate=sp.limit(sp.log(g_sym)/_n,_n,sp.oo)
                    if g_exp_rate.is_finite:
                        g_base=float(sp.exp(g_exp_rate).evalf())
                        if abs(g_base-max_mod)<0.05:
                            bs=_clean_exp(max_mod)
                            big_o=f"\\Theta(n \\cdot {bs}^n)" if abs(max_mod-round(max_mod))<1e-6 else f"\\Theta(n \\cdot ({bs})^n)"
                            steps.append(DemoStep(title="线性递推 — 共振检测",description="",
                                latex=f"\\text{{特征根底数 }}{max_mod:.6g}\\text{{ 与 }}g(n)\\text{{ 共振}}\\Rightarrow{big_o}"))
                            return big_o,steps,None
                except: pass
                big_o=homo
            elif g_b>0: big_o=_fmt_theta(g_b)
            else: big_o=homo
        except: big_o=homo
    else: big_o=homo
    steps.append(DemoStep(title="线性递推 — 最终结论", description="", latex=f"T(n)={big_o}"))
    return big_o,steps,None


# ═══════════════ STRATEGY 3: DOMAIN SUBSTITUTION ═══════════════

def _domain_substitution(terms, g_sym: sp.Expr) -> Optional[Tuple[str, List[DemoStep], Optional[str]]]:
    from parse_clean import _sanitize
    steps = []
    has_nl = any('sqrt' in t.function or '**' in t.function.replace('^','**') for t in terms)
    if not has_nl: return None
    steps.append(DemoStep(title="换元构造法 — 识别非线性",
        description="检测到 $\\sqrt{n}$ 等，令 $n=2^m$。", latex="n=2^m,\\;S(m)=T(2^m)"))
    new_terms = []
    for t in terms:
        fs = _sanitize(t.function.strip()).replace('^','**')
        fs = re.sub(r'sqrt\s*\(\s*n\s*\)','2**(m/2)',fs)
        fs = re.sub(r'n\s*\*\*\s*0\.5','2**(m/2)',fs)
        from main import TermItem
        new_terms.append(TermItem(coefficient=t.coefficient, function=fs))
    simplified = False
    for t in new_terms:
        mt = re.search(r'2\s*\*\*\s*\(\s*m\s*/\s*(\d+)\s*\)', t.function)
        if mt: t.function = f"n/{int(mt.group(1))}"; simplified = True
    if not simplified: return None
    g_m = sp.simplify(g_sym.subs(_n,2**_m).subs(_m,_n))
    steps.append(DemoStep(title="换元构造法 — 代换完成",
        description="$n=2^m$ 代入得标准递推。",
        latex=f"S(n)={' + '.join(f'{_display_coeff(t.coefficient)} S({t.function})' for t in new_terms)}+{sp.latex(g_m)}"))
    strat,params,ks=_classify(new_terms)
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

def _continuous_approx(terms, g_sym: sp.Expr) -> Optional[Tuple[str, List[DemoStep], Optional[str]]]:
    if len(terms)!=1: return None
    t=terms[0]; c=_is_sub(t.function)
    # Also handle variable-step: T(n - log n), T(n - sqrt n), etc.
    if c is None:
        expr = _parse_safe(t.function)
        if expr is not None and expr.has(_n) and expr != _n:
            # Try to extract asymptotic step: diff = n - f(n) ≈ ?
            try:
                diff = sp.limit(_n - expr, _n, sp.oo)
                if diff.is_number and diff > 0: c = float(diff)
                else:
                    # Variable step: approximate dT/dn ≈ g(n) / step(n)
                    from parse_clean import _n as pn
                    c = 1.0  # will use the actual step in the DE
            except: c = None
        if c is None: return None
    k = _parse_coeff(t.coefficient)
    if abs(k - 1.0) > 1e-9: return None
    steps=[DemoStep(title="连续化微积分 — 启动",
        description=f"$T(n)-T(f(n))\\approx g(n)$，连续化处理。",
        latex=f"T(n)-T({sp.latex(_parse_safe(t.function))})\\approx {sp.latex(g_sym)}")]
    T_func=sp.Function('T')
    # If step is variable, use n - f(n) as step in derivative
    if c is not None and _is_sub(t.function) is not None:
        step_expr = sp.Integer(int(c))
    else:
        step_expr = _n - _parse_safe(t.function)
    de=sp.Eq(sp.Derivative(T_func(_n),_n), g_sym / step_expr)
    steps.append(DemoStep(title="连续化微积分 — 微分方程",
        description=f"$dT/dn\\approx g(n) / (n - f(n))$。", latex=sp.latex(de)))
    try:
        sol=sp.dsolve(de,T_func(_n))
        if sol is not None:
            steps.append(DemoStep(title="连续化微积分 — dsolve",
                description="SymPy 求得通解。", latex=sp.latex(sol)))
            big_o=_big_o_from_expr(sp.simplify(sol.rhs))
            steps.append(DemoStep(title="连续化微积分 — 渐近阶",
                description="提取主导项。", latex=f"T(n)\\approx{big_o}"))
            return big_o,steps,None
    except Exception: pass
    steps.append(DemoStep(title="连续化微积分 — 失败", description="", latex=""))
    return None


# ═══════════════ STRATEGY 5: POINCARÉ-PERRON ═══════════════

def _poincare_perron(terms, g_sym: sp.Expr) -> Optional[Tuple[str, List[DemoStep], Optional[str]]]:
    """Detect variable coefficients or variable f_i(n) forms. Take limits to approximate."""
    from parse_clean import _sanitize
    steps = []
    has_var_coeff = False; has_var_func = False

    # Check coefficients for variable forms
    for t in terms:
        cs = _sanitize(t.coefficient.strip())
        expr = _parse_safe(cs)
        if expr is not None and expr.has(_n) and not expr.is_number:
            has_var_coeff = True; break

    # Check functions for non-standard forms
    for t in terms:
        expr = _parse_safe(t.function)
        if expr is not None and expr.has(_n) and expr != _n:
            if _is_div(t.function) is None and _is_sub(t.function) is None:
                has_var_func = True; break

    if not has_var_coeff and not has_var_func:
        return None

    steps.append(DemoStep(title="庞加莱-佩隆 — 启动",
        description="检测到非常系数或非标准变元，$n\\to\\infty$ 极限分析。",
        latex="\\text{分析系数与变元的极限行为}"))

    # Take limits of coefficients as n→∞
    lim_ks = []
    for t in terms:
        cs = _sanitize(t.coefficient.strip())
        expr = _parse_safe(cs)
        if expr is not None and expr.has(_n):
            try:
                ratio = sp.limit(expr, _n, sp.oo)
                if ratio.is_number: lim_ks.append(float(ratio))
                else: lim_ks.append(float(expr.subs(_n, 1e6).evalf()))
            except: lim_ks.append(_parse_coeff(cs))
        else: lim_ks.append(_parse_coeff(cs))

    # Take limits of f_i(n)/n as n→∞
    lim_ratios = []
    for t in terms:
        expr = _parse_safe(t.function)
        if expr is None: continue
        try:
            ratio = sp.limit(expr / _n, _n, sp.oo)
            if ratio.is_number: lim_ratios.append(float(ratio))
            else: lim_ratios.append(None)
        except: lim_ratios.append(None)

    steps.append(DemoStep(title="庞加莱-佩隆 — 极限参数",
        description=f"系数极限: {', '.join(_clean_exp(k) for k in lim_ks)}。",
        latex=f"k_i\\to {', '.join(_clean_exp(k) for k in lim_ks)}"))

    # Classify limiting behavior
    if all(r is not None for r in lim_ratios):
        all_div = all(0 < r < 1 for r in lim_ratios)
        all_sub = all(abs(r - 1) < 1e-6 for r in lim_ratios)
        if all_div:
            b_lim = [1.0 / r for r in lim_ratios]
            r = _akra_bazzi(b_lim, lim_ks, g_sym)
            if r: big_o, ab_steps, note = r; steps.extend(ab_steps); return big_o, steps, note
        elif all_sub:
            c_lim = [1 for _ in lim_ratios]
            r = _linear_recurrence(c_lim, lim_ks, g_sym)
            if r: big_o, ln_steps, note = r; steps.extend(ln_steps); return big_o, steps, note

    # If only coefficients were variable, retry with limiting coefficients
    if has_var_coeff and not has_var_func and lim_ks:
        strat, params, ks_orig = _classify(terms)
        if strat == "division":
            r = _akra_bazzi(params, lim_ks, g_sym)
            if r:
                big_o, ab_steps, note = r
                steps.append(DemoStep(title="庞加莱-佩隆 — 极限 Akra-Bazzi",
                    description="系数取 $n\\to\\infty$ 极限后求解。", latex=""))
                steps.extend(ab_steps); return big_o, steps, note
        elif strat == "subtraction":
            r = _linear_recurrence(params, lim_ks, g_sym)
            if r:
                big_o, ln_steps, note = r
                steps.extend(ln_steps); return big_o, steps, note

    return None


# ═══════════════ STRATEGY 6: NUMERICAL ═══════════════

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

def _numerical_solve(terms, g_sym: sp.Expr) -> Optional[Tuple[str, List[DemoStep], Optional[str]]]:
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
    if float(np.max(tv)-np.min(tv))<1e-12:
        steps.append(DemoStep(title="数值模拟 — 序列恒常",
            description="全为常数。", latex=f"T(n)\\equiv{tv[0]:.6g}"))
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
    steps.append(DemoStep(title="数值模拟 — 拟合对比",
        description="", latex=f"\\begin{{array}}{{{'c'*len(results_table)}}} {rows_latex} \\end{{array}}"))
    steps.append(DemoStep(title="数值模拟 — 最佳匹配", description="",
        latex=f"\\text{{最佳: }}{best_name},\\;R^2={best_r2:.4f}"))
    steps.append(DemoStep(title="提示", description="数值结果建议交叉验证。", latex=""))
    return best_name,steps,"由于符号数学限制，此结果由数值模拟曲线拟合得出"
