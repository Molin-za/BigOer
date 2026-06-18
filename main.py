"""
BigOer v4 — 通用算法时间复杂度分析平台
6+1 Strategy Chain: MasterThm → AkraBazzi → LinearRec → DomainSub
                   → ContinuousApprox → PoincarePerron → NumericalFit
"""
import sys; sys.dont_write_bytecode = True

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from parse_clean import (_n, _sympy_expr, _parse_coeff, _classify,
    _preprocess_terms, _build_formula, _has_variable_coeff)
from solve_engine import (DemoStep, _master_theorem, _akra_bazzi,
    _linear_recurrence, _domain_substitution, _continuous_approx,
    _poincare_perron, _numerical_solve)

# ── App
app = FastAPI(title="BigOer", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ── Models
class TermItem(BaseModel):
    coefficient: str; function: str

class SolveRequest(BaseModel):
    terms: List[TermItem]; g_n: str

class SolveResponse(BaseModel):
    success: bool; latex_formula: str; final_complexity: str
    demo_steps: List[DemoStep]; method: str; note: Optional[str] = None


# ── Strategy Chain
def _try_all_strategies(req: SolveRequest, g_sym, strategy: str, params: list, ks: list):
    # Pre-check: variable coefficients → try Poincaré-Perron first
    if _has_variable_coeff(req.terms):
        r = _poincare_perron(req.terms, g_sym)
        if r and r[0]: return (*r, "庞加莱-佩隆")

    all_pos = all(k > 1e-12 for k in ks) if ks else False

    if all_pos and strategy in ("division",) and len(params) == 1:
        r = _master_theorem(ks[0], params[0], g_sym)
        if r and r[0]: return (*r, "主定理")

    if all_pos and strategy == "division" and params:
        r = _akra_bazzi(params, ks, g_sym)
        if r and r[0]: return (*r, "Akra-Bazzi 定理")

    if strategy in ("subtraction",):
        if len(params) == 1:
            r = _continuous_approx(req.terms, g_sym)
            if r and r[0]: return (*r, "连续化微积分")
        r = _linear_recurrence(params, ks, g_sym)
        if r and r[0]: return (*r, "线性特征方程")
    else:
        r = _continuous_approx(req.terms, g_sym)
        if r and r[0]: return (*r, "连续化微积分")

    r = _domain_substitution(req.terms, g_sym)
    if r and r[0]: return (*r, "换元构造法")

    r = _poincare_perron(req.terms, g_sym)
    if r and r[0]: return (*r, "庞加莱-佩隆")

    r = _numerical_solve(req.terms, g_sym)
    if r and r[0]: return (*r, "数值模拟曲线拟合")
    return None


# ── Endpoint
@app.post("/api/solve", response_model=SolveResponse)
async def solve(req: SolveRequest):
    if not req.terms: raise HTTPException(400, "至少需要一项递推项")
    latex_formula = _build_formula(req)
    try: g_sym = _sympy_expr(req.g_n)
    except ValueError as e: raise HTTPException(400, str(e))

    cleaned_terms, g_sym, pre_err = _preprocess_terms(req.terms, g_sym)
    if pre_err: raise HTTPException(400, pre_err)
    if all(abs(_parse_coeff(t.coefficient)) < 1e-12 for t in cleaned_terms):
        raise HTTPException(400, "所有递推系数均为零")
    if cleaned_terms != req.terms:
        latex_formula = _build_formula(SolveRequest(terms=cleaned_terms, g_n=req.g_n))

    strategy, params, ks = _classify(cleaned_terms)
    result = _try_all_strategies(
        SolveRequest(terms=cleaned_terms, g_n=req.g_n), g_sym, strategy, params, ks)
    if result is None: raise HTTPException(500, "所有策略均求解失败")
    big_o, steps, note, method = result

    return SolveResponse(success=True, latex_formula=latex_formula,
        final_complexity=big_o, demo_steps=steps, method=method, note=note)


@app.get("/api/health")
async def health(): return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8000)
