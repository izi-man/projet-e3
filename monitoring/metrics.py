@app.get("/metrics")
def get_metrics():

    total = metrics["successful_predictions"]

    if total > 0:
        average_response_time = (
            metrics["total_response_time_ms"] / total
        )

        low_confidence_rate = (
            metrics["low_confidence_predictions"] / total
        ) * 100
    else:
        average_response_time = 0
        low_confidence_rate = 0

    alert = None

    if low_confidence_rate > 50:
        alert = (
            "ALERTE : plus de 50% des prédictions "
            "ont une faible confiance."
        )

        logger.warning(alert)

    return {
        "total_requests": metrics["total_requests"],
        "successful_predictions": metrics["successful_predictions"],
        "errors": metrics["errors"],
        "average_response_time_ms": round(
            average_response_time,
            2
        ),
        "low_confidence_predictions": metrics[
            "low_confidence_predictions"
        ],
        "low_confidence_rate_percent": round(
            low_confidence_rate,
            2
        ),
        "alert": alert
    }