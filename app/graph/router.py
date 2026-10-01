def verification_router(state):
    if state.get("verified", False):
        return "finish"
    retry_count = state.get(
        "retry_count",
        0
    )
    max_retries = state.get(
        "max_retries",
        2
    )
    if retry_count >= max_retries:
        return "finish"
    rewritten_query = state.get(
        "verification",
        {}
    ).get(
        "rewritten_query",
        ""
    )
    if not rewritten_query.strip():
        return "finish"

    return "retry"