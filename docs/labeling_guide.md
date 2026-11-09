# Prompt Labeling Guide

Every prompt used to train or evaluate Lonn's complexity scorer is labeled with
the rules below, whether a person or a model labels it. When a rule and your gut
disagree, follow the rule and note the prompt so the rule can be fixed.

Each prompt gets two labels:

1. **Topic distribution**: a probability for each of the four topics, summing to 1.
2. **Complexity**: a whole number from 1 to 5.

---

## 1. Topics

Label the **task the user wants done**, not the subject the prompt mentions.
"Write a poem about Python" is creative writing, not coding.

| Topic | The user wants... | Includes |
|---|---|---|
| `coding` | code written, fixed, explained, reviewed or designed | programs, scripts, SQL, regex, shell commands, config files, APIs, error messages and stack traces, software architecture, explaining how an algorithm or library works |
| `math_reasoning` | an answer that must be worked out step by step | calculations, equations, proofs, statistics, word problems, logic puzzles, estimates, probability, multi-step planning with a correct answer |
| `creative_writing` | text where style, tone or voice is the point | stories, poems, jokes, song ideas, marketing copy, emails and letters where tone matters, rewriting, paraphrasing, editing, translating, role-play, naming and slogan brainstorming |
| `general_knowledge_extraction` | information: facts, explanations, advice or content pulled from given text | factual questions, explaining concepts, how-to (non-code), advice, recommendations, comparisons, summarizing, extracting or classifying information from provided text, casual chat |

### Borderline rules

| Situation | Label |
|---|---|
| Maths needed to write code ("function that computes compound interest") | `coding` |
| Code needed to do maths ("solve this integral with SymPy") | `coding` 0.6, `math_reasoning` 0.4 |
| Explaining a software concept without writing code ("how does a hash map work?") | `coding` |
| Explaining a non-software concept ("how do vaccines work?") | `general_knowledge_extraction` |
| Informative article or blog post ("write a blog post explaining inflation") | `general_knowledge_extraction` 0.6, `creative_writing` 0.4 |
| Professional email or cover letter | `creative_writing` |
| Summarizing, or answering questions about, pasted text | `general_knowledge_extraction` |
| Rewriting or improving pasted text | `creative_writing` |
| Greeting or small talk ("hi", "how are you?") | `general_knowledge_extraction` |
| Mechanical text changes with one right answer (remove vowels, change case, strip punctuation) | `general_knowledge_extraction` |
| Fixing grammar or spelling in a sentence or email | `creative_writing` |
| Translation | `creative_writing` |
| Computer or network troubleshooting (drivers, Wi-Fi, reinstalling an OS) | `general_knowledge_extraction`, unless the user wants code or commands written, then `coding` |
| Word and letter puzzles, riddles | `math_reasoning` |
| Continuing a story or role-play the user has started | `creative_writing` |
| Competitive programming problems | `coding`; add `math_reasoning` 0.3 or 0.4 when the core difficulty is mathematical |
| Jailbreak or role-play framing ("act as my grandmother who...") | Label the task actually requested |

### Mixed prompts

Use a split only when the user clearly asks for two kinds of work, for example
"write a story that teaches how binary search works, with code". Rules:

- At most two topics above zero.
- Use steps of 0.1. The main task gets at least 0.5.
- If one part is only a small side request, give it 0.2 or 0.3.
- A single-topic prompt gets 1.0 on that topic. Do not spread small amounts
  "just in case".

---

## 2. Complexity

**Complexity is how difficult it is to produce a good answer to the prompt.**
It describes the prompt only. Deciding what to do with that, such as which model
to use, is the router's job, so do not let routing considerations affect a label.

| Level | Name | What a good answer takes | Typical signs |
|---|---|---|---|
| 1 | Trivial | Almost nothing; hard to get wrong | One fact, one step, greeting, simple conversion, rephrasing one sentence |
| 2 | Simple | One clear piece of work | One clear task, common knowledge, short output, no traps |
| 3 | Moderate | Some care; easy to get partly wrong | Several steps, some domain knowledge, a few constraints, a medium amount of context to use |
| 4 | Hard | Careful reasoning or real expertise | Multi-step reasoning, many constraints, long context that must be used carefully, subtle bugs |
| 5 | Very hard | Expert-level work that may still fall short | Research or competition level, large multi-part designs, highly constrained long outputs |

### Rules

- **Judge the task, not the prompt length.** "Prove there are infinitely many
  primes" is short but a 3. A pasted 2,000-word article with "summarize this" is
  a 2.
- **Creativity alone does not make a task hard.** A poem about autumn is a 2. A
  sonnet in iambic pentameter, with an acrostic, in the voice of a named author
  is a 4.
- **Constraints add difficulty.** Each hard requirement on format, length,
  style or correctness pushes the level up.
- **Context adds difficulty only if it must be used carefully.** Finding one
  fact in pasted text is easy; reconciling details across a long document is not.
- **Unclear or unanswerable prompts:** rate the task as the user meant it.
- **When torn between two levels, pick the one whose "typical signs" match
  more of the prompt.** Compare with the worked examples below. If it is still
  a tie, pick the lower level, so that everyone breaks ties the same way.

---

## 3. Worked examples

| Prompt | Topic | Complexity |
|---|---|---|
| hi! | `general_knowledge_extraction` 1.0 | 1 |
| What is the capital of Australia? | `general_knowledge_extraction` 1.0 | 1 |
| Convert 5 miles to kilometers | `math_reasoning` 1.0 | 1 |
| Write a Python function that reverses a string | `coding` 1.0 | 1 |
| Write a short poem about the sea at night | `creative_writing` 1.0 | 2 |
| Summarize this paragraph: [150 words] | `general_knowledge_extraction` 1.0 | 2 |
| Why does my SQL LEFT JOIN return duplicate rows? | `coding` 1.0 | 2 |
| Rewrite this email so it sounds friendlier: [short email] | `creative_writing` 1.0 | 2 |
| A train leaves at 3pm at 80 km/h, another at 4pm at 100 km/h. When does the second catch up? | `math_reasoning` 1.0 | 2 |
| Explain how vaccines train the immune system | `general_knowledge_extraction` 1.0 | 2 |
| Here is my 80-line Django view that times out under load: [code]. Find the problem and fix it. | `coding` 1.0 | 3 |
| Write a cover letter for this job ad, matching my CV: [ad + CV] | `creative_writing` 1.0 | 3 |
| Solve this system of three equations and explain each step | `math_reasoning` 1.0 | 3 |
| Tell me a story about writing the perfect database schema in Python | `creative_writing` 0.6, `coding` 0.4 | 3 |
| Compare the monetary policies of the ECB and the Fed since 2020 and their effects | `general_knowledge_extraction` 1.0 | 3 |
| Design a database schema for a multi-tenant SaaS app with row-level security and write the migrations | `coding` 1.0 | 4 |
| Prove that the square root of 2 is irrational, then generalize to any non-square integer | `math_reasoning` 1.0 | 4 |
| Write a 1,000-word short story in second person, present tense, with no letter "e" | `creative_writing` 1.0 | 4 |
| Read these three contracts and list every clause where they conflict: [long text] | `general_knowledge_extraction` 1.0 | 4 |
| Implement a lock-free concurrent hash map in Rust and argue why it is linearizable | `coding` 1.0 | 5 |
| Solve this International Mathematical Olympiad problem with a full proof: [problem] | `math_reasoning` 1.0 | 5 |

---

## 4. File format

Labels are stored as JSON Lines, one prompt per line:

```json
{"prompt": "Write a short poem about the sea at night", "topics": {"creative_writing": 1.0}, "complexity": 2, "source": "handwritten", "labeler": "joel"}
```

- `topics` lists only topics above zero; the others are 0.
- `source` records where the prompt came from (for example `wildchat`, `dolly`,
  `gsm8k`, `synthetic`).
- `labeler` records who labeled it (a person's name, or `claude`).
- Test-set files also get `"reviewed_by": "<name>"` once a person has checked them.
