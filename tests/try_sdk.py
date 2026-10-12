from lonnrouter import LonnRouter, Signal

with LonnRouter(base_url="http://127.0.0.1:8000") as router:
    print("Health:", router.health().status)
    print("Models:", [model.id for model in router.models()])

    # Checks routing without loading the classifier or calling a provider.
    selection = router.route(
        "Write a sorting function",
        signal=Signal(complexity=0.5, topic_scores={"coding": 1.0}),
    )
    print("Selected:", selection.model.id)

    # Calls the selected model; requires NVIDIA credentials on the server.
    result = router.generate("What is 17 * 23?", max_tokens=256)
    print(result.content)
    print("Finish reason:", result.finish_reason)