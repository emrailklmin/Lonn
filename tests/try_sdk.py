from lonnrouter import LonnRouter, Signal

with LonnRouter(base_url="http://127.0.0.1:8000") as router:
    print("Health:", router.health().status)
    print("Models:", [model.id for model in router.models()])

    # Checks routing without the task classifier or a provider call.
    selection = router.route("Write a sorting function", signal=Signal(task_scores={"code_function": 1.0}))
    print("Selected:", selection.model.id, selection.estimates[0])

    # Automatic classification (downloads the embedding model on first use), still no provider call.
    selection = router.route("Write a sorting function", lam=40)
    print("Classified:", max(selection.signal.task_scores.items(), key=lambda item: item[1]))

    # Calls the selected model; requires the provider key on the server.
    result = router.generate("What is 17 * 23?", max_tokens=256)
    print(result.content)
    print("Finish reason:", result.finish_reason)
