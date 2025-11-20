def _percent(value: float) -> float:
    try:
        return round(float(value) * 100.0, 1)
    except Exception:
        return None


CLASSIFICATION_STORIES = {
    "LogisticRegression": "Prefers clean signal and keeps decisions linear, perfect when you want a transparent baseline.",
    "KNN": "Looks for look-alike rows to vote on the answer, handy when similar people or items behave alike.",
    "DecisionTree": "Builds an easy-to-explain flow of yes/no questions, so teams can reason about the outcome.",
    "SVC": "Focuses on the most decisive examples and carves a tidy boundary between classes.",
    "RandomForest": "Blends dozens of shallow trees, giving you steady accuracy even when some features are noisy.",
    "GradientBoosting": "Learns in small corrective steps so it can capture tricky, non-linear patterns without overreacting.",
}

REGRESSION_STORIES = {
    "LinearRegression": "Keeps things simple and fast by drawing a straight trend line through your data.",
    "PolynomialRegression": "Extends the straight line into gentle curves, helping follow wavy real-world signals.",
    "Ridge": "Adds a safety belt to linear regression so no single column can hijack the predictions.",
    "Lasso": "Finds the most impactful columns and quietly mutes the rest, which helps with lean feature sets.",
    "ElasticNet": "Balances Ridge and Lasso so you get stability plus feature selection at the same time.",
    "KNN": "Predicts by averaging the closest past rows, great when nearby examples behave similarly.",
    "DecisionTree": "Splits the data into understandable rules, which makes communicating predictions easier.",
    "SVR": "Fits a smooth tube through the data, ignoring small wiggles to focus on the bigger movement.",
    "LinearSVR": "Keeps the SVR idea but trims it down for faster, large-scale runs.",
    "RandomForest": "Averages many small trees to stay robust against outliers and column noise.",
    "GradientBoosting": "Stacks small improvements to chase subtle relationships without needing manual feature work.",
}


def _classification_metric_text(metric: float | None) -> str:
    pct = _percent(metric)
    if pct is None:
        return ""
    return f"It stayed on target for roughly {pct} out of 100 rows in our evaluation batch."


def _regression_metric_text(metric: float | None) -> str:
    pct = _percent(metric)
    if pct is None:
        return ""
    return f"It explained about {pct}% of the swings we saw in your target column."


def describe_model(task_type: str, model_name: str, metric: float | None) -> dict:
    """
    Returns a dict with a friendly explanation and metric blurb for the given model.
    """
    task_type = (task_type or "").lower()

    if task_type == "classification":
        story = CLASSIFICATION_STORIES.get(
            model_name,
            "This classifier offered the best balance of confidence and stability for your dataset.",
        )
        metric_text = _classification_metric_text(metric)
    else:
        story = REGRESSION_STORIES.get(
            model_name,
            "This regressor captured the overall trend most reliably during evaluation.",
        )
        metric_text = _regression_metric_text(metric)

    explanation = story if not metric_text else f"{story} {metric_text}"

    return {
        "explanation": explanation,
        "metric_text": metric_text or None,
    }

