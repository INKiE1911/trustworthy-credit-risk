"""Step 12: counterfactuals, "what would have to change for a better decision?".

Only actionable fields change: a smaller loan (amount and annuity together, i.e. borrow less)
or a higher income. Age, gender and the external scores are never touched (actionable recourse,
Ustun et al. 2019). Every candidate goes through ScoringService.build_features, so all ratios and
cross-table features are rebuilt with the training code. A library like DiCE would instead change
the 240 model inputs directly (for example "BUR_DEBT_TO_INCOME"), which nobody can act on.

    from creditrisk.explain.counterfactual import counterfactuals
    counterfactuals(service, applicant_id, target=0.0355)   # target: probability to get below
"""

from pydantic import ValidationError

from creditrisk.serving.service import Changes, ScoringService

# option name -> (what-if fields scaled together, factors tried from the smallest change up)
OPTIONS = {
    "smaller loan": (("loan_amount", "annuity"), [round(1 - 0.05 * k, 2) for k in range(1, 19)]),
    "higher income": (("income",), [round(1 + 0.1 * k, 1) for k in range(1, 31)]),
}


def counterfactuals(service: ScoringService, applicant_id: int, target: float) -> list[dict]:
    """For each option, the smallest change on its grid that brings the probability below target.

    Returns [] if the applicant is already below target. factor is None when no change on the
    grid is enough. The grid search is exact for this model (no monotonicity is assumed).
    """
    current = service.profile(applicant_id).current

    def probability(changes: Changes) -> float:
        X = service.build_features(applicant_id, changes)
        return float(service.model.predict_proba(X)[0, 1])

    before = probability(Changes())
    if before < target:
        return []
    found = []
    for option, (fields, factors) in OPTIONS.items():
        if any(current[f] is None for f in fields):
            continue
        result = {"option": option, "factor": None, "changes": {}, "probability": None,
                  "probability_before": before}
        for factor in factors:
            try:
                changes = Changes(**{f: current[f] * factor for f in fields})
            except ValidationError:  # past the allowed input range
                break
            p = probability(changes)
            if p < target:
                result.update(factor=factor, probability=p,
                              changes=changes.model_dump(exclude_none=True))
                break
        found.append(result)
    return found
