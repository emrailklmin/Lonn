# Lonn Prototype Architecture

The Lonn prototype consists of four core responsibilities:

1. Analyse the incoming prompt.
2. Produce a complexity score and topic probability distribution.
3. Deterministically select the most suitable model.
4. Execute the request and return the result to the web application.

---

# 1. System Architecture

```mermaid
flowchart LR

    %% =========================
    %% WEB APPLICATION
    %% =========================

    subgraph WEB["Web App — examples/app.py"]

        USER["User"]

        UI["Web App UI
        ----------------
        submit_prompt(prompt)
        display_execution_state()
        display_response()"]

    end


    %% =========================
    %% Lonn CORE
    %% =========================

    subgraph CORE["Lonn Core — src/lonn"]

        COMPLEXITY["Complexity Scorer
        complexity.py
        ----------------
        INPUT:
        prompt: str

        OUTPUT:
        ComplexityResult"]

        ROUTER["Deterministic Router
        router.py
        ----------------
        INPUT:
        ComplexityResult
        list[Model]

        OUTPUT:
        Selected Model"]

        EXECUTOR["Model Executor
        executor.py
        ----------------
        INPUT:
        prompt: str
        model: Model

        OUTPUT:
        ModelResponse"]

        MODELPOOL["Model Pool
        model_pool.py
        ----------------
        INPUT:
        config/models.yaml

        OUTPUT:
        list[Model]"]

    end


    %% =========================
    %% CONFIGURATION
    %% =========================

    CONFIG["config/models.yaml
    ----------------
    Model metadata
    Provider
    Token cost
    Context window
    Topic aptitude"]


    %% =========================
    %% EXTERNAL PROVIDER
    %% =========================

    PROVIDER["External LLM Provider
    ----------------
    OpenAI
    Anthropic
    Google
    etc."]


    %% =========================
    %% FLOW
    %% =========================

    USER -->|"Prompt"| UI

    UI -->|"prompt: str"| COMPLEXITY

    COMPLEXITY -->|"ComplexityResult"| ROUTER

    CONFIG --> MODELPOOL

    MODELPOOL -->|"list[Model]"| ROUTER

    ROUTER -->|"Selected Model"| EXECUTOR

    UI -->|"Original prompt"| EXECUTOR

    EXECUTOR -->|"API Request"| PROVIDER

    PROVIDER -->|"LLM Response + usage"| EXECUTOR

    EXECUTOR -->|"ModelResponse"| UI

    UI -->|"Runtime flow + final response"| USER
```

---

# 2. Core Classes and Data Contracts

```mermaid
classDiagram

    class ComplexityResult {
        +float complexity
        +dict~TopicCategory,float~ topic_scores
    }

    class TopicCategory {
        <<enumeration>>
        CODING
        MATH_REASONING
        CREATIVE_WRITING
        GENERAL_KNOWLEDGE_EXTRACTION
    }

    class ComplexityScorer {
        +score_prompt(prompt: str) ComplexityResult
    }


    class Model {
        +str id
        +str provider

        +float cost_per_m_input
        +float cost_per_m_output

        +int context_window

        +dict~TopicCategory,float~ topic_aptitudes
    }


    class ModelPool {
        +load_model_pool(path: str) list~Model~
    }


    class Router {
        +select_model(
            signal: ComplexityResult,
            models: list~Model~
        ) Model
    }


    class TokenUsage {
        +int prompt_tokens
        +int completion_tokens
        +int total_tokens

        +float input_cost
        +float output_cost
        +float total_cost
    }


    class ModelResponse {
        +str content
        +Model model
        +TokenUsage usage
        +float latency_ms
        +str finish_reason
    }


    class ModelExecutor {
        +execute(
            prompt: str,
            model: Model
        ) ModelResponse
    }


    class WebApp {
        +submit_prompt(prompt: str)
        +display_execution_state()
        +display_response(response: ModelResponse)
    }


    ComplexityScorer --> ComplexityResult : produces

    ComplexityResult --> TopicCategory : contains

    Model --> TopicCategory : aptitude per topic

    ModelPool --> Model : loads

    Router --> ComplexityResult : uses

    Router --> Model : selects

    ModelExecutor --> Model : executes

    ModelExecutor --> ModelResponse : produces

    ModelResponse --> TokenUsage : contains

    ModelResponse --> Model : identifies

    WebApp --> ComplexityScorer : calls

    WebApp --> Router : observes result

    WebApp --> ModelExecutor : receives response
```

---

# 3. Data Structures

## Complexity Result

The complexity scorer produces two different types of information.

### Overall complexity

```text
complexity ∈ [0, 1]
```

Example:

```python
complexity = 0.72
```

### Topic distribution

`topic_scores` represents a probability distribution over the supported
categories.

Therefore:

```text
Σ topic_scores = 1.0
```

Example:

```python
topic_scores = {
    "coding": 0.35,
    "math_reasoning": 0.45,
    "creative_writing": 0.08,
    "general_knowledge_extraction": 0.12,
}
```

The values sum to:

```text
0.35 + 0.45 + 0.08 + 0.12 = 1.00
```

---

# 4. Model Configuration

Model topic values are **aptitude scores**, not probability distributions.

Each aptitude is independently defined between:

```text
0.0 <= aptitude <= 1.0
```

They therefore do **not** have to sum to 1.

The single model configuration is `config/models.yaml`. It contains a YAML
list with the same field names as `Model`: `id`, `provider`,
`cost_per_m_input`, `cost_per_m_output`, `context_window`, and `topic_aptitudes`.
Full LiteLLM IDs use `<provider>/<publisher>/<model>`, where `provider` is
`nvidia_nim` or `openrouter` and must match the entry's `provider` field.
OpenRouter is used only as a credit-based way to reach models; its own routing
is not used.

Additional configuration fields are `active`, `price_basis`, and
`matched_provider_key` (the source of a third-party price estimate). The loader
filters inactive entries and validates the shared contract. Pricing references
are metadata and must never replace the execution ID.

Discovery (`scripts/discover_models.py`) saves every catalog ID
without requiring pricing or guessing chat support from names. Availability
probing (`scripts/check_models.py`) separately checks text generation, records
timestamps and attempt outcomes, and never mutates routing configuration.
Timeouts and rate limits are transient failures; tool capability is separate.
A catalog listing is not a successful inference check. Generated JSON reports
are written to the ignored `reports/` directory and are not model configuration.

`load_model_pool(path)` validates all active entries and returns `list[Model]`.
It rejects duplicate IDs, unknown context limits, incomplete aptitude maps,
invalid prices, and missing price provenance (`price_basis`). It never invents
aptitudes or treats missing metadata as zero. A pool with no active models fails
explicitly. Configuration may retain inactive incomplete candidates.

The pool contains documented context values and benchmark-derived aptitude
proxies. Sources, raw results, normalization, and deployment limitations are
recorded in [model_metadata.md](model_metadata.md). These are initial routing
estimates, not measured performance of our hosted requests. Prices sourced
from other providers remain prototype cost estimates, not NVIDIA billing rates.

---

# 5. Runtime Execution Flow

```mermaid
sequenceDiagram

    actor User

    participant UI as Web App
    participant CS as Complexity Scorer
    participant MP as Model Pool
    participant R as Router
    participant E as Executor
    participant LLM as LLM Provider


    User->>UI: Submit prompt

    UI->>CS: score_prompt(prompt)

    activate CS

    CS-->>UI: ComplexityResult

    deactivate CS


    Note over UI,CS: complexity = 0.72

    Note over UI,CS: topic_scores sum to 1.0


    UI->>MP: load_model_pool()

    activate MP

    MP-->>UI: list[Model]

    deactivate MP


    UI->>R: select_model(ComplexityResult, models)

    activate R

    R-->>UI: Selected Model

    deactivate R


    Note over UI,R: Web app displays selected model


    UI->>E: execute(prompt, selected_model)

    activate E


    E->>LLM: API request


    activate LLM

    LLM-->>E: content + token usage

    deactivate LLM


    E-->>UI: ModelResponse

    deactivate E


    UI-->>User: Display response


    Note over UI,User: UI also displays runtime execution details
```

---

# 6. Runtime Web App

The local Streamlit app in `examples/app.py` exposes the internal routing process rather than
only showing the final LLM response.

Run with `python3 -m streamlit run examples/app.py --server.address localhost`.
Automatic mode uses Laya; manual mode supplies an explicit topic/complexity for
isolated routing and API tests. Analyze-only mode makes no NVIDIA request.
Submitting a live request invokes scorer → router → executor; changing controls
does not resend it. Results and provider-error categories are shown in the UI.

A request progresses through:

```mermaid
flowchart LR

    A["1. Prompt Received"]

    B["2. Complexity Analysis"]

    C["3. Topic Distribution"]

    D["4. Model Routing"]

    E["5. Model Selected"]

    F["6. Model Execution"]

    G["7. Response Received"]

    H["8. Usage & Cost"]

    I["9. Final Response"]


    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
```

The web app can therefore display information such as:

- Prompt token count
- Complexity score
- Topic probability distribution
- Candidate models
- Selected model
- Model topic aptitudes
- Context window
- Input token price
- Output token price
- Execution latency
- Input tokens
- Output tokens
- Total request cost
- Final model response

---

# 7. Responsibility Boundaries

## Prototype routing policy

`router.select_model(signal, models, alpha=0.5)` (also available as
`Router().select_model(signal, models, alpha=0.5)`) returns a model from the pool.

1. Compute quality as `sum(topic_probability * model_topic_aptitude)`.
   Missing aptitudes count as zero.
2. Add `cost_per_m_input` and `cost_per_m_output`, both prices per million
   tokens in the same currency. This is an equal-token price proxy, not an
   actual request cost estimate.
3. Normalize cost by dividing by the highest combined price in the pool.
   When all models are free, every normalized cost is zero.
4. Maximize `alpha * quality - (1 - alpha) * normalized_cost`.
5. Break score ties by lower price, higher quality, then `(provider, id)`.

`weighted_score(quality, cost, alpha=0.5)` exposes the weighted function.
All three arguments must be finite and in [0, 1]. Alpha 0 prioritizes cost;
alpha 1 prioritizes quality. The default is 0.5. Pool-relative normalization
puts cost on the same [0, 1] scale as quality, but adding/removing models can
change the tradeoff. Raw price units do not affect selection.

Complexity is still validated as part of the input contract, but no longer
imposes a quality threshold. Callers may explicitly pass
`alpha=signal.complexity` if they want complexity to control the tradeoff.
Context-window filtering still requires prompt tokens and an output budget.

An empty pool, duplicate `(provider, id)` entries, invalid probabilities,
aptitudes, complexity, alpha, or prices raise `ValueError`. Probabilities must
sum to one within an absolute tolerance of `1e-6`.

Run the router tests with `python3 tests/test_router.py` or
`python3 -m unittest discover -s tests` from the repository root.

## Executor policy and shared contracts

`schemas.py` owns the shared dataclasses: `ComplexityResult`, `Model`,
`TokenUsage`, and `ModelResponse`. A dataclass supplies generated initialization,
representation, and equality methods; it does not validate field types itself.
The router and executor consume these contracts, and the app displays them.

`executor.execute(prompt, model, timeout=120, max_tokens=4096)` and
`ModelExecutor().execute(...)` send a non-streaming text request using LiteLLM's
NVIDIA NIM or OpenRouter provider, chosen by the model's `provider`. The
executor loads the repository `.env` without overriding existing environment
values and requires only the key for that provider: `NVIDIA_NIM_API_KEY` or
`OPENROUTER_API_KEY`.

The result contains text, the selected model, measured request latency in
milliseconds, token counts, estimated costs, and `finish_reason`. Cost is
`tokens / 1_000_000 * configured_price` for input and output separately.
A `length` finish reason allows the app to identify a truncated response.

Invalid inputs, missing credentials, missing usage, or non-text responses fail
explicitly. Provider exceptions propagate to the caller. The executor performs
no rerouting, automatic retries, streaming, or tool execution. Its output budget
must be below the context window; full prompt-plus-output context validation
remains a future task requiring model-specific token counting.

Dependencies are listed in `requirements.txt`; executor tests use mocked calls.

```mermaid
flowchart TB

    A["complexity.py
    Understand the request"]

    B["router.py
    Decide which model should handle it"]

    C["model_pool.py
    Describe which models are available"]

    D["executor.py
    Execute the selected model"]

    E["app.py
    Visualise the entire process"]


    A --> B

    C --> B

    B --> D

    A -.runtime state.-> E

    B -.runtime state.-> E

    D -.runtime state.-> E
```

The modules should remain independent:

- `complexity.py` does not know how routing works.
- `router.py` does not call model APIs.
- `model_pool.py` does not make routing decisions.
- `executor.py` does not determine which model is best.
- `app.py` orchestrates the prototype and visualises what happens at runtime.
