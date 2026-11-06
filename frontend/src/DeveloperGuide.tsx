import { useState } from "react";
import { ArrowLeft, ArrowRight, BookOpen, Check, Copy } from "lucide-react";
import { Button } from "./components/ui/button";
import type { Account } from "./types";
import { PythonCode } from "./components/PythonCode";

type Chapter = {
  title: string;
  kicker: string;
  intro: string;
  cells: { title: string; text: string; code?: string; note?: string }[];
};
export function chapters(application: string, endpoint: string): Chapter[] {
  const setup = `import os\nfrom lonnrouter import LonnRouter\n\nclient = LonnRouter(\n    api_key=os.environ["LONN_API_KEY"],\n    application="${application}",\n    base_url=${JSON.stringify(endpoint)},\n)`;
  return [
    {
      title: "Quickstart",
      kicker: "01 / FIRST REQUEST",
      intro:
        "Connect your application to Lönn in a few lines. The SDK sends requests to your local Lönn server, which handles routing and the provider connection.",
      cells: [
        {
          title: "Install the SDK",
          text: "From the repository root, install the Python package into your environment.",
          code: "python -m pip install ./sdk/python",
        },
        {
          title: "Create a client and generate",
          text: "Set LONN_API_KEY in your environment first. Replace the application name with one you can access from Projects & apps.",
          code: `${setup}\n\nwith client:\n    result = client.generate("Explain binary search.")\n    print(result.content)\n    print("Model:", result.model.id)`,
          note: "For an application key, omit the application argument. Configure the presenter’s OpenRouter credential in portal Settings before making live requests.",
        },
      ],
    },
    {
      title: "Intelligent Routing",
      kicker: "02 / ROUTING",
      intro:
        "One integration can use multiple models. Lönn classifies each prompt and selects from the eligible pool using the existing task router.",
      cells: [
        {
          title: "Route and generate",
          text: "generate() routes automatically; there is no router name to supply in this SDK.",
          code: `${setup}\n\nwith client:\n    result = client.generate("Write a function to group words by length.")\n    print(result.content)\n    print("Selected:", result.model.id)\n    print("Effective λ:", result.lam)`,
        },
        {
          title: "Inspect the decision first",
          text: "route() ranks candidates without making a provider request.",
          code: `${setup}\n\nwith client:\n    decision = client.route("Summarize the key points.")\n    for candidate in decision.models:\n        print(candidate.id)`,
          note: "Automatic analysis may download the local classifier weights on first use.",
        },
      ],
    },
    {
      title: "Lambda Preferences",
      kicker: "03 / ROUTING CONTROL",
      intro:
        "λ controls the trade-off in the router’s score. Higher values favor quality; lower values favor cost.",
      cells: [
        {
          title: "Set a request preference",
          text: "Pass a literal lam between 0 and 1: 0 is cost only, 1 is quality only. The default is 0.50. When no administrator has enforced a higher-level policy, it applies to this request.",
          code: `${setup}\n\nwith client:\n    result = client.generate("Explain a neural network.", lam=0.50)\n    print("Effective λ:", result.lam, result.lambda_source)\n    print(result.content)`,
          note: "Organization, team, project, and application enforcement takes precedence over a request value. Otherwise your request and personal preference take precedence over unlocked application and project defaults; the fallback is 0.50.",
        },
      ],
    },
    {
      title: "Direct Model Selection",
      kicker: "04 / EXPLICIT MODEL",
      intro:
        "For a specific task, choose a model explicitly instead of letting Lönn choose among candidates. Your organization and team restrictions still apply.",
      cells: [
        {
          title: "Choose an eligible model",
          text: "List eligible models first, then pass one model ID to generate().",
          code: `${setup}\n\nwith client:\n    eligible = client.models()\n    if not eligible:\n        raise RuntimeError("No eligible models for this application")\n    result = client.generate(\n        "Explain the water cycle.",\n        model=eligible[0].id,\n    )\n    print(result.content)`,
          note: "Direct selection makes a single model attempt; it does not fall back to a different model. An excluded or unknown model is rejected before a provider call.",
        },
      ],
    },
    {
      title: "Model Pool",
      kicker: "05 / CONFIGURATION",
      intro:
        "The configured pool contains active, measured OpenRouter models allowed by organization and team model selections.",
      cells: [
        {
          title: "List the configured pool",
          text: "model_pool() lists pool members before privacy restrictions are applied.",
          code: `${setup}\n\nwith client:\n    for model in client.model_pool():\n        print(model.id, model.provider)\n        print("Input / 1M:", model.cost_per_m_input)\n        print("Output / 1M:", model.cost_per_m_output)`,
          note: "Configured prices are model token rates. Customer charges also include the demo service fee.",
        },
      ],
    },
    {
      title: "Eligible Models",
      kicker: "06 / POLICY",
      intro:
        "Eligibility applies your security requirements to demo model assumptions. Provider safeguards and processing locations are not verified or controlled by the demo.",
      cells: [
        {
          title: "List models you can use now",
          text: "models() returns the eligible models for the key and application. An empty list means no current model satisfies the policy.",
          code: `${setup}\n\nwith client:\n    for model in client.models():\n        print(model.id)`,
          note: "Eligibility is enforced on the server for automatic routing, fallback, and direct selection. Inspect exclusion reasons in the portal’s Model pool page.",
        },
      ],
    },
    {
      title: "Applications & Usage",
      kicker: "07 / ATTRIBUTION",
      intro:
        "Personal keys identify a user and require an authorized application. Application keys are bound to one application and do not need an application argument.",
      cells: [
        {
          title: "Use an application key",
          text: "Create or copy a key from Projects & apps. The request is attributed to that application, project, and team.",
          code: `import os\nfrom lonnrouter import LonnRouter\n\nwith LonnRouter(api_key=os.environ["LONN_APP_KEY"]) as client:\n    result = client.generate("Draft a short project update.")\n    print(result.content)\n    print("Customer cost:", result.total_cost_usd)`,
          note: "The dashboard updates from the existing usage log. Use its organization, team, project, application, and user filters to explore costs and requests.",
        },
      ],
    },
    {
      title: "FAQ & Troubleshooting",
      kicker: "08 / REFERENCE",
      intro: "A few checks for the most common integration issues.",
      cells: [
        {
          title: "Authentication & access",
          text: "401 means a missing, invalid, or revoked key. Personal keys also need an application; verify that your user is assigned to it. Application keys cannot select another application.",
        },
        {
          title: "Models & policies",
          text: "If the eligible list is empty or direct selection returns 403, review the organization, team, and privacy restrictions. An enforced λ overrides your requested value; check result.lam or Projects & apps.",
        },
        {
          title: "Provider errors & costs",
          text: "A 502, 503, 429, or 504 can indicate provider configuration, unavailable models, rate limits, or timeouts. The dashboard records successful and failed attempts; result.total_cost_usd includes provider costs plus a Lönn fee of 20% of positive estimated savings against the highest-quality eligible model. result.billing exposes the breakdown.",
        },
        {
          title: "Handle an API error",
          text: "The SDK exposes the HTTP status without returning raw provider details.",
          code: `import os\nfrom lonnrouter import LonnAPIError, LonnRouter\n\ntry:\n    with LonnRouter(\n        api_key=os.environ["LONN_API_KEY"],\n        application="${application}",\n    ) as client:\n        result = client.generate("Hello!")\n        print(result.content)\nexcept LonnAPIError as error:\n    print("Lönn API status:", error.status_code)`,
        },
      ],
    },
  ];
}
function CodeCell({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="guide-code">
      <div className="guide-code-header">
        <span>
          <i /> {code.startsWith("python -m") ? "TERMINAL" : "PYTHON"}
        </span>
        <button
          onClick={async () => {
            try {
              await navigator.clipboard.writeText(code);
              setCopied(true);
              setTimeout(() => setCopied(false), 1600);
            } catch {
              setCopied(false);
            }
          }}
        >
          {copied ? <Check size={14} /> : <Copy size={14} />}{" "}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre>
        <code>
          <PythonCode code={code} />
        </code>
      </pre>
    </div>
  );
}
export function DeveloperGuide({ account }: { account: Account }) {
  const [chapter, setChapter] = useState(0);
  const pages = chapters(
    account.user.role === "private_user" ? "personal-app" : "engineering-app",
    account.endpoint,
  );
  const current = pages[chapter];
  return (
    <div className="guide-layout">
      <nav className="guide-nav" aria-label="Guide chapters">
        <div className="guide-nav-top">
          <BookOpen size={16} />
          <strong>Developer guide</strong>
        </div>
        <small>GETTING STARTED</small>
        {pages.map((p, i) => (
          <button
            key={p.title}
            className={i === chapter ? "guide-chapter active" : "guide-chapter"}
            aria-current={i === chapter ? "page" : undefined}
            onClick={() => setChapter(i)}
          >
            <span>{String(i + 1).padStart(2, "0")}</span>
            {p.title}
          </button>
        ))}
        <div className="guide-nav-foot">
          LÖNN AI <span>·</span> LOCAL DEMO
        </div>
      </nav>
      <article className="guide-page">
        <div className="guide-heading">
          <div>
            <p className="eyebrow">{current.kicker}</p>
            <h2>{current.title}</h2>
            <p>{current.intro}</p>
          </div>
          <span className="badge">
            CHAPTER {chapter + 1} / {pages.length}
          </span>
        </div>
        {current.cells.map((cell, i) => (
          <section className="guide-cell" key={cell.title}>
            <div className="guide-cell-number">
              {String(i + 1).padStart(2, "0")}
            </div>
            <div className="guide-cell-content">
              <h3>{cell.title}</h3>
              <p>{cell.text}</p>
              {cell.code && <CodeCell code={cell.code} />}{" "}
              {cell.note && <div className="guide-note">{cell.note}</div>}
            </div>
          </section>
        ))}
        <div className="guide-pagination">
          <Button
            variant="outline"
            disabled={chapter === 0}
            onClick={() => setChapter(chapter - 1)}
          >
            <ArrowLeft size={15} /> Previous
          </Button>
          <span>
            {chapter + 1} / {pages.length}
          </span>
          <Button
            disabled={chapter === pages.length - 1}
            onClick={() => setChapter(chapter + 1)}
          >
            Next chapter <ArrowRight size={15} />
          </Button>
        </div>
      </article>
    </div>
  );
}
