"""The FastAPI service. Run `make api`, then open http://localhost:8000/docs to try it.

POST /score              {"applicant_id": <id>, "changes": {"income": 250000}}
                         -> probability of default, vs the average, top reasons both ways
GET  /applicants         demo applicant ids
GET  /applicants/{id}    the applicant's current values for the 8 what-if fields
GET  /health             is the model loaded?
"""

from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query

from creditrisk.serving.service import Profile, ScoreRequest, ScoreResponse, ScoringService

app = FastAPI(
    title="Trustworthy credit risk API",
    version="0.1.0",
    description="Default probability and reasons for real (unlabelled) Home Credit applicants. "
                "Probabilities are not calibrated yet (Step 10).",
)


@lru_cache(maxsize=1)
def get_service() -> ScoringService:
    """Load the model and demo data once, on the first request."""
    return ScoringService()


@app.get("/health")
def health() -> dict:
    service = get_service()
    return {"status": "ok", "model": service.model_name,
            "applicants": len(service.applicant_ids)}


@app.get("/applicants")
def applicants(limit: int = Query(50, ge=1, le=5000)) -> dict:
    ids = get_service().applicant_ids
    return {"count": len(ids), "applicant_ids": ids[:limit]}


@app.get("/applicants/{applicant_id}", response_model=Profile)
def applicant(applicant_id: int) -> Profile:
    try:
        return get_service().profile(applicant_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown applicant {applicant_id}") from None


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    try:
        return get_service().score(request.applicant_id, request.changes)
    except KeyError:
        raise HTTPException(status_code=404,
                            detail=f"Unknown applicant {request.applicant_id}") from None
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
