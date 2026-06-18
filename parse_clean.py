"""BigOer — parsing, sanitization, classification, preprocessing."""

import re, math, ast
from typing import List, Optional, Tuple

import sympy as sp
import numpy as np

# ── Global SymPy symbols ──
_n = sp.symbols('n', positive=True, integer=True)
_x = sp.symbols('x', positive=True, real=True)
_p = sp.symbols('p', real=True)
_r = sp.symbols('r')
_m = sp.symbols('m', positive=True)

# ═══════════════ SANITIZE & PARSE ═══════════════

def _sanitize(s: str) -> str:
    s = s.strip()
    s = s.replace('（', '(').replace('）', ')')
    s = re.sub(r'(sqrt|log|sin|cos|tan|exp|abs)\s+\(', r'\1(', s)
    s = re.sub(r'(sqrt|log|sin|cos|tan|exp|abs)\s+([a-zA-Z0-9]+)', r'\1(\2)', s)
    s = re.sub(r'(?<=[a-zA-Z0-9])\s+(?=\))', '', s)
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
    """Parse coefficient string → numeric. Uses ast.literal_eval for safety."""
    s = s.strip()
    try: return float(ast.literal_eval(s))
    except: pass
    expr = _parse_safe(s)
    if expr is None: return 1.0
    if not expr.has(_n): return float(expr.evalf())
    try:
        lim = sp.limit(expr, _n, sp.oo)
        if lim.is_number: return float(lim.evalf())
    except: pass
    try:
        f = sp.lambdify(_n, expr, 'numpy'); v = f(1000.0)
        if np.isfinite(v): return float(v)
    except: pass
    return 1.0


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


def _classify(terms) -> Tuple[str, list, list]:
    d_c, s_c = 0, 0; params, ks = [], []
    for t in terms:
        k = _parse_coeff(t.coefficient)
        if abs(k) < 1e-12: continue
        b, c = _is_div(t.function), _is_sub(t.function)
        if b is not None: params.append(b); ks.append(k); d_c += 1
        elif c is not None: params.append(c); ks.append(k); s_c += 1
        else: return "mixed", [], [_parse_coeff(x.coefficient) for x in terms if abs(_parse_coeff(x.coefficient))>=1e-12]
    if s_c == 0: return "division", params, ks
    if d_c == 0: return "subtraction", params, ks
    return "mixed", params, ks


def _has_variable_coeff(terms) -> bool:
    """Check if any coefficient depends on n (variable coefficient)."""
    for t in terms:
        cs = _sanitize(t.coefficient.strip())
        expr = _parse_safe(cs)
        if expr is not None and expr.has(_n) and not expr.is_number:
            return True
    return False


# ═══════════════ PREPROCESSING ═══════════════

def _preprocess_terms(terms, g_sym: sp.Expr):
    tn_coeff = 0.0; other = []
    for t in terms:
        fs = t.function.strip()
        if fs == 'n' or re.match(r'^\s*n\s*\*\*\s*1\s*$', fs) or re.match(r'^\s*n\s*\^\s*1\s*$', fs):
            tn_coeff += _parse_coeff(t.coefficient)
        else: other.append(t)
    if abs(tn_coeff) < 1e-12: return terms, g_sym, ""
    left = 1.0 - tn_coeff
    if abs(left) < 1e-12: return [], g_sym, "方程系数冲突退化: 两侧 T(n) 抵消，无法构成有效递推关系"
    from main import TermItem
    norm_terms = [TermItem(coefficient=str(_parse_coeff(t.coefficient)/left), function=t.function) for t in other]
    norm_g = sp.simplify(g_sym / left)
    return norm_terms, norm_g, ""


# ═══════════════ DISPLAY HELPERS ═══════════════

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


def _display_coeff(s: str) -> str:
    s = s.strip()
    try: return _clean_coeff(float(ast.literal_eval(s)))
    except: pass
    expr = _parse_safe(s)
    if expr is not None and not expr.has(_n): return _clean_coeff(float(expr.evalf()))
    if expr is not None: return sp.latex(expr)
    return s


def _usr_latex(s: str) -> str:
    s = s.strip()
    s = re.sub(r'(\w+|\))\^(\w+|\([^)]+\))', r'\1^{\2}', s)
    s = re.sub(r'\*', r' \\cdot ', s)
    s = re.sub(r'\blog\b', r'\\log', s); s = re.sub(r'\bsqrt\b', r'\\sqrt', s)
    return s


def _build_formula(req) -> str:
    parts = []
    for t in req.terms:
        c = _display_coeff(t.coefficient)
        f = _sanitize(t.function).replace('**','^').replace('*',' \\cdot ')
        f = re.sub(r'\blog\b', r'\\log', f); f = re.sub(r'\bsqrt\b', r'\\sqrt', f)
        parts.append(f"T({f})" if c == '1' else f"{c} T({f})")
    gs = _sanitize(req.g_n).replace('**','^').replace('*',' \\cdot ')
    gs = re.sub(r'\blog\b', r'\\log', gs); gs = re.sub(r'\bsqrt\b', r'\\sqrt', gs)
    return "T(n) = " + (" + ".join(parts) if parts else "0") + " + " + gs


# ═══════════════ NUMERICAL HELPERS ═══════════════

def _eval_fn_str(f_str: str, n_val: float) -> float:
    s = f_str.strip()
    m = re.match(r'^\s*n\s*/\s*(\d+\.?\d*)\s*$', s)
    if m: return math.floor(n_val / float(m.group(1)))
    m = re.match(r'^\s*n\s*-\s*(\d+\.?\d*)\s*$', s)
    if m: return max(0.0, n_val - float(m.group(1)))
    s = s.replace('^', '**')
    try:
        expr = sp.sympify(s, {'n': n_val, 'log': sp.log, 'sqrt': sp.sqrt})
        v = float(expr.evalf())
        return math.floor(v) if v != int(v) else v
    except: return n_val


def _compute_seq(terms, g_func, N: int = 1000) -> np.ndarray:
    T = np.zeros(N + 1, dtype=np.float64); T[0], T[1] = 0.0, 1.0
    for i in range(2, N + 1):
        val = 0.0
        for t in terms:
            fi = _eval_fn_str(t.function, float(i))
            idx = int(max(0, min(i - 1, math.floor(fi))))
            val += _parse_coeff(t.coefficient) * T[idx]
        val += g_func(float(i))
        if not np.isfinite(val) or abs(val) > 1e300: val = 1e300
        T[i] = val
    return T
