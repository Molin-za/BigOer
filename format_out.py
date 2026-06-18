"""BigOer — LaTeX formatters, Big-O extraction, asymptotic detection."""

import re, math
from typing import Tuple

import sympy as sp
import numpy as np
from parse_clean import _n, _clean_coeff

# ═══════════════ NUMBER FORMATTERS ═══════════════

def _clean_exp(e: float) -> str:
    if abs(e) < 1e-6: return "0"
    if abs(e - round(e)) < 1e-6: return str(int(round(e)))
    s = f"{e:.4f}".rstrip('0').rstrip('.'); return s


def _symbolic_log(a_num: float, b_num: float) -> str:
    if abs(b_num - 1.0) < 1e-9: return _clean_exp(math.log(a_num) / math.log(b_num))
    a_i = int(a_num) if abs(a_num - int(a_num)) < 1e-9 else None
    b_i = int(b_num) if abs(b_num - int(b_num)) < 1e-9 else None
    if a_i is None or b_i is None or a_i <= 0 or b_i <= 1:
        return _clean_exp(math.log(a_num) / math.log(b_num))
    for base in range(2, min(max(a_i, b_i) + 1, 50)):
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


# ═══════════════ LaTeX COMPLEXITY CONSTRUCTORS ═══════════════

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


# ═══════════════ ASYMPTOTIC DETECTION ═══════════════

def _detect_log_k(g_sym, p_val: float) -> Tuple[int, int]:
    """Detect log power k: g(n)=Θ(n^p·log^k n). Uses 3-point log-log slope.
    Returns (log_k, case) where case 1=subdominant, 2=resonance, 3=dominant."""
    try:
        _f0 = sp.lambdify(_n, g_sym / _n**p_val, 'numpy')
        v0 = [_f0(ni) for ni in [1e4, 1e6, 1e8]]
        v0 = [abs(x) for x in v0 if np.isfinite(x)]
        if len(v0) < 2: return 0, 2
        v_slope = np.log(v0[-1]/max(v0[0],1e-12)) / np.log(1e8/1e4) if v0[0]>1e-12 else 0
        if v_slope < -0.15: return 0, 1  # truly subdominant (polynomial gap)
        if -0.15 <= v_slope < -0.02: return 0, 0  # ambiguous: let Akra-Bazzi handle it
        if v_slope > 0.15:  # truly dominant
            for k in range(1, 5):
                _fk = sp.lambdify(_n, g_sym / (_n**p_val * sp.log(_n)**k), 'numpy')
                vk = [abs(_fk(ni)) for ni in [1e4, 1e6, 1e8]]
                vk = [x for x in vk if np.isfinite(x)]
                if len(vk) >= 2:
                    max_vk, min_vk = max(vk), min(vk)
                    if max_vk > 0.1 and min_vk / max_vk > 0.6: return k, 2
                    if vk[-1] < 0.001 * max(vk[0], 1e-9): return k-1, 2
            return 0, 3
        v0r = v0[-1] / max(v0[0], 1e-12) if v0[0] > 1e-12 else 99
        if 0.7 < v0r < 1.4: return 0, 2
        for k in range(1, 5):
            _fk = sp.lambdify(_n, g_sym / (_n**p_val * sp.log(_n)**k), 'numpy')
            vk = [abs(_fk(ni)) for ni in [1e4, 1e6, 1e8]]
            vk = [x for x in vk if np.isfinite(x)]
            if len(vk) >= 2:
                max_vk, min_vk = max(vk), min(vk)
                if max_vk > 0.1 and min_vk / max_vk > 0.6: return k, 2
                if vk[-1] < 0.001 * max(vk[0], 1e-9): return k-1, 2
        return 0, 2
    except: return 0, 0


def _back_substitute(big_o: str) -> str:
    """Convert S(m)→T(n)=S(log_2 n).  n→log n, n^p→(log n)^p, log n→log log n."""
    V = '\uE000V\uE000'
    result = re.sub(r'(?<!\\)n\^\{', '('+V+')^{', big_o)
    result = result.replace('\\log n', '\\log '+V)
    result = re.sub(r'(?<!\\)(?<!\w)n(?!\w)', V, result)
    result = result.replace(V, '\\log n')
    result = re.sub(r'(\\log\s+n)\s+(\\log)', r'\1 \\cdot \2', result)
    return result


# ═══════════════ BIG-O FROM EXPRESSION ═══════════════

def _big_o_from_expr(expr: sp.Expr) -> str:
    """Extract clean Big-O from a SymPy expression using numeric convergence."""
    simp = sp.simplify(expr)
    for s in list(simp.free_symbols):
        if s != _n: simp = simp.subs(s, 1)
    simp = sp.simplify(simp)

    # Candidates ordered fast-growing → slow-growing
    for label, cand, n_small, n_large in [
        ("O(n \\cdot 3^n)",_n*3**_n, 10, 30), ("O(n \\cdot 2^n)",_n*2**_n, 10, 30),
        ("O(3^n)",3**_n, 10, 30), ("O(2^n)",2**_n, 10, 30), ("O(n!)",sp.factorial(_n), 5, 10),
        ("O(n^3)",_n**3, 1e3, 1e5), ("O(n^2)",_n**2, 1e3, 1e5),
        ("O(n \\log^2 n)",_n*sp.log(_n)**2, 1e3, 1e5), ("O(n \\log n)",_n*sp.log(_n), 1e3, 1e5),
        ("O(n)",_n, 1e3, 1e5),
        ("O(\\log^2 n)",sp.log(_n)**2, 1e3, 1e5), ("O(\\log n)",sp.log(_n), 1e3, 1e5),
        ("O(\\log \\log n)",sp.log(sp.log(_n)), 1e3, 1e5),
        ("O(1)",1, 1e3, 1e5),
    ]:
        try:
            _fc = sp.lambdify(_n, simp/cand, 'numpy')
            v1, v2 = _fc(n_small), _fc(n_large)
            if np.isfinite(v1) and np.isfinite(v2):
                r = abs(v2) / max(abs(v1), 1e-12)
                if label == "O(1)" and r > 1.05: continue
                if 0.75 < r < 1.35 and abs(v2) > 0.001: return label
        except: pass

    # Handle li(n) → n/log n
    if 'li(' in str(simp) or 'li(' in sp.latex(simp):
        return "O\\left(\\frac{n}{\\log n}\\right)"

    # Polynomial detection via log-log slope
    try:
        _fp = sp.lambdify(_n, simp, 'numpy')
        vp1, vp2 = abs(_fp(1e6)), abs(_fp(1e8))
        if np.isfinite(vp2) and vp2 > 1e-6 and vp1 > 1e-6:
            pv = np.log(vp2/vp1) / np.log(1e8/1e6)
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
                            if np.isfinite(w2) and 0.75<r<1.35 and 0.001<w2<1000: log_k=k; break
                            elif np.isfinite(w2) and w2<0.01: log_k=k-1; break
                        except: continue
                elif np.isfinite(u02) and 0.75 < u02/max(u01,1e-9) < 1.35: log_k=0
                # If log_k=0 but the expression seems slightly subdominant, check log-log factor
                if log_k == 0 and u02 < 0.9:
                    try:
                        _fll = sp.lambdify(_n, simp/(_n**pv * sp.log(sp.log(_n))), 'numpy')
                        wll1, wll2 = abs(_fll(1e6)), abs(_fll(1e8))
                        rll = wll2/max(wll1,1e-12) if wll1>1e-12 else 99
                        if np.isfinite(wll2) and 0.75 < rll < 1.35 and abs(wll2) > 0.001:
                            return f"O(n^{{{_clean_exp(pv)}}} \\log \\log n)"
                    except: pass
                return _fmt_theta(pv, log_k, prefix="O")
    except: pass

    # Exponential detection via growth rate
    try:
        _fe = sp.lambdify(_n, simp, 'numpy')
        ve1, ve2 = abs(_fe(500)), abs(_fe(1000))
        if np.isfinite(ve2) and ve2 > 1e-6 and ve1 > 1e-6:
            gr = np.log(ve2/ve1) / 500; be = np.exp(gr)
            if be > 1.05:
                _fr = sp.lambdify(_n, simp/(_n*be**_n), 'numpy')
                vr1, vr2 = abs(_fr(500)), abs(_fr(1000))
                rr = vr2/max(vr1,1e-12) if vr1>1e-12 else 99
                if np.isfinite(vr2) and 0.75<rr<1.35 and 0.001<vr2<1000:
                    return f"O(n \\cdot {_clean_exp(be)}^n)"
                return _fmt_exp(be, prefix="O")
    except: pass

    return f"O\\left({sp.latex(simp)}\\right)"
