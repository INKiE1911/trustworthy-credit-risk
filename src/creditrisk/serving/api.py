"""The FastAPI service. Run `make api`, then open http://localhost:8000/docs to try it.

POST /score              {"applicant_id": <id>, "changes": {"income": 250000}}
                         -> probability of default, decision (approve / refer / decline),
                            reason codes, top reasons both ways
GET  /applicants         demo applicant ids
GET  /applicants/{id}    the applicant's current values for the 8 what-if fields
GET  /applicants/{id}/counterfactuals
                         the smallest smaller-loan / higher-income change that reaches the
                         next better decision (decline -> refer -> approve)
GET  /health             is the model loaded?
"""

from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query

from creditrisk.explain.counterfactual import next_decision
from creditrisk.serving.service import (
    ImpossibleChange,
    Profile,
    ScoreRequest,
    ScoreResponse,
    ScoringService,
    UnknownApplicant,
)

app = FastAPI(
    title="Trustworthy credit risk API",
    version="0.2.0",
    description="Default probability, decision and reasons for demo Home Credit applicants. "
                "Probabilities are calibrated (Step 10); approve / refer / decline comes from "
                "the conformal layer (Step 11).",
)


@lru_cache(maxsize=1)
def get_service() -> ScoringService:
    """Load the model and demo data once, on the first request."""
    return ScoringService()


@app.get("/health")
def health() -> dict:
    service = get_service()
    return {"status": "ok", "model": service.model_name,
            "applicants": len(service.applicant_ids), "synthetic": service.synthetic,
            "demo_model": service.demo_model}


@app.get("/applicants")
def applicants(limit: int = Query(50, ge=1, le=5000)) -> dict:
    ids = get_service().applicant_ids
    return {"count": len(ids), "applicant_ids": ids[:limit]}


@app.get("/applicants/{applicant_id}", response_model=Profile)
def applicant(applicant_id: int) -> Profile:
    try:
        return get_service().profile(applicant_id)
    except UnknownApplicant:
        raise HTTPException(status_code=404, detail=f"Unknown applicant {applicant_id}") from None


@app.get("/applicants/{applicant_id}/counterfactuals")
def counterfactual(applicant_id: int) -> dict:
    try:
        return next_decision(get_service(), applicant_id)
    except UnknownApplicant:
        raise HTTPException(status_code=404, detail=f"Unknown applicant {applicant_id}") from None


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    try:
        return get_service().score(request.applicant_id, request.changes)
    except UnknownApplicant:
        raise HTTPException(status_code=404,
                            detail=f"Unknown applicant {request.applicant_id}") from None
    except ImpossibleChange as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
