import "./CustomerApp.css";
import React, { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  ArrowUp,
  ArrowUpRight,
  Check,
  ChevronDown,
  Code2,
  FileCode2,
  MessageSquare,
  Link2,
  LockKeyhole,
  Plus,
  RefreshCw,
  Sparkles,
  X,
} from "lucide-react";
import { Button } from "./components/ui/button";
import { CopyButton } from "./components/CopyButton";
import { PythonCode } from "./components/PythonCode";
import { RuntimeRouting, type RuntimeTrace } from "./components/RuntimeRouting";
import { LonnLogo } from "./components/LonnLogo";
import {
  DEFAULT_LAMBDA,
  MAX_LAMBDA,
  LambdaSlider,
} from "./components/LambdaSlider";
import { api, shortModel, money, type Account, type Billing } from "./types";

type Connection = {
  application: { id: string; name: string };
  key_type: "personal" | "application";
  preference_scope: string;
  lam: number;
  source: string;
  locked: boolean;
};
const preferenceKey = (connection: Connection) =>
  `lonn.lambda.${connection.preference_scope}.${connection.application.id}`;
function savedLambda(connection: Connection): number | null {
  try {
    const value = localStorage.getItem(preferenceKey(connection));
    if (value === null) return null;
    const number = Number(value);
    return Number.isFinite(number) && number >= 0 && number <= MAX_LAMBDA ? number : null;
  } catch { return null; }
}
type Model = { id: string; provider: string };
type Message = {
  prompt: string;
  content?: string;
  model?: string;
  mode?: "router" | "direct";
  cost?: number;
  billing?: Billing;
  lam?: number;
  lambdaSource?: string;
  lambdaLocked?: boolean;
  finish?: string;
  error?: string;
};

export function CustomerApp() {
  const [key, setKey] = useState(""),
    [application, setApplication] = useState("");
  const [draftKey, setDraftKey] = useState(""),
    [draftApplication, setDraftApplication] = useState("");
  const [connection, setConnection] = useState<Connection | null>(null);
  const [endpoint, setEndpoint] = useState(window.location.origin);
  const [models, setModels] = useState<Model[]>([]),
    [connecting, setConnecting] = useState(false);
  const [connectionError, setConnectionError] = useState(""),
    [menu, setMenu] = useState<"connection" | "routing" | null>(null);
  const [mode, setMode] = useState<"router" | "direct">("router"),
    [model, setModel] = useState("");
  const [lam, setLam] = useState(DEFAULT_LAMBDA),
    [customLam, setCustomLam] = useState(false);
  const [prompt, setPrompt] = useState(""),
    [busy, setBusy] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]),
    [showCode, setShowCode] = useState(true);
  const [trace, setTrace] = useState<RuntimeTrace | null>(null);
  const connectionRef = useRef<HTMLDivElement>(null),
    routingRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null),
    inputRef = useRef<HTMLTextAreaElement>(null);
  const connectionButtonRef = useRef<HTMLButtonElement>(null),
    routingButtonRef = useRef<HTMLButtonElement>(null);

  function closeMenu() {
    setMenu(null);
    (menu === "connection"
      ? connectionButtonRef
      : routingButtonRef
    ).current?.focus();
  }

  const headers = (apiKey: string, app: string) => ({
    Authorization: `Bearer ${apiKey}`,
    ...(app ? { "X-Lonn-Application": app } : {}),
  });
  async function connect(apiKey: string, app: string, refresh = false) {
    setConnecting(true);
    setConnectionError("");
    try {
      const options = { headers: headers(apiKey, app) };
      const [context, eligible] = await Promise.all([
        api<Connection>("/connection", options),
        api<Model[]>("/models", options),
      ]);
      setKey(apiKey);
      setApplication(app);
      setDraftKey(apiKey);
      setDraftApplication(app);
      setConnection(context);
      setModels(eligible);
      setModel((previous) =>
        refresh && eligible.some((m) => m.id === previous)
          ? previous
          : (eligible[0]?.id ?? ""),
      );
      const saved = savedLambda(context);
      setLam(context.locked ? context.lam : (saved ?? context.lam));
      setCustomLam(!context.locked && saved !== null);
      if (!refresh) setMenu(null);
    } catch (e) {
      if (refresh) setConnection(null);
      setConnectionError(
        e instanceof Error ? e.message : "Could not connect to Lönn",
      );
    } finally {
      setConnecting(false);
    }
  }
  function chooseLambda(value: number) {
    setLam(value);
    setCustomLam(true);
    if (connection) {
      try { localStorage.setItem(preferenceKey(connection), String(value)); } catch { /* Storage is optional. */ }
    }
  }
  useEffect(() => {
    let active = true;
    Promise.all([
      api<Account>("/api/account"),
      api<{ applications: { name: string }[] }>("/api/hierarchy"),
    ])
      .then(([account, hierarchy]) => {
        if (active) setEndpoint(account.endpoint);
        if (active && account.api_key && hierarchy.applications.length) {
          void connect(account.api_key, hierarchy.applications[0].name);
        }
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, []);
  useEffect(() => {
    const close = (event: PointerEvent) => {
      if (
        !connectionRef.current?.contains(event.target as Node) &&
        !routingRef.current?.contains(event.target as Node)
      )
        setMenu(null);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && menu) closeMenu();
    };
    document.addEventListener("pointerdown", close);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", close);
      document.removeEventListener("keydown", escape);
    };
  }, [menu]);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [messages]);

  async function send(e: React.FormEvent) {
    e.preventDefault();
    if (
      busy ||
      connecting ||
      !connection ||
      !prompt.trim() ||
      !models.length ||
      (mode === "direct" && !model)
    )
      return;
    const text = prompt.trim();
    setTrace({ status: "pending", mode });
    setBusy(true);
    setMenu(null);
    setPrompt("");
    setMessages((previous) => [...previous, { prompt: text }]);
    try {
      const routed = await api<{
        models: { id: string; context_window: number }[];
        signal: { task_scores: Record<string, number>; complexity: number | null };
        estimates: { model_id: string; score: number }[];
        lam: number;
      }>("/route", {
        method: "POST",
        headers: headers(key, application),
        body: JSON.stringify({
          prompt: text,
          ...(mode === "router" ? { router: "router1.0", lam } : {}),
        }),
      });
      const selected = mode === "router"
        ? routed.models.slice(0, 5).find((candidate) => candidate.context_window > 4096)?.id
        : routed.models.find((candidate) => candidate.id === shortModel(model))?.id;
      setTrace({
        status: "generating", mode, lam: routed.lam,
        topics: routed.signal.task_scores, eligible: routed.models.map((candidate) => candidate.id),
        ranking: mode === "router" ? routed.estimates : undefined, selected,
      });
      const result = await api<{
        model: { id: string };
        lam: number;
        lambda_source: string;
        lam_locked: boolean;
        total_cost_usd: number;
        billing?: Billing;
        response: { content: string; finish_reason: string };
        signal: { task_scores: Record<string, number> };
        eligible_models: string[];
        attempted_models: string[];
        estimates: { model_id: string; score: number }[];
        attempts: { model_id: string; check: string }[];
        escalated: boolean;
      }>("/generate", {
        method: "POST",
        headers: headers(key, application),
        body: JSON.stringify({
          prompt: text,
          signal: routed.signal,
          ...(mode === "direct"
            ? { model: shortModel(model) }
            : { router: "router1.0", lam }),
        }),
      });
      setTrace({
        status: "complete", mode, lam: result.lam,
        topics: result.signal.task_scores, eligible: result.eligible_models,
        ranking: mode === "router" ? result.estimates : undefined,
        selected: result.model.id, attempts: result.attempts,
        attemptedModels: result.attempted_models, escalated: result.escalated,
      });
      setMessages((previous) => [
        ...previous.slice(0, -1),
        {
          prompt: text,
          content: result.response.content || "No final text returned.",
          model: result.model.id,
          mode,
          cost: result.total_cost_usd,
          billing: result.billing,
          lam: result.lam,
          lambdaSource: result.lambda_source,
          lambdaLocked: result.lam_locked,
          finish: result.response.finish_reason,
        },
      ]);
      // The policy can change while the connection menu is open. Trust the
      // request's server snapshot for both its receipt and the next request.
      setConnection(
        (previous) =>
          previous && {
            ...previous,
            lam: result.lam,
            source: result.lambda_source,
            locked: result.lam_locked,
          },
      );
      if (mode === "router" || result.lam_locked) setLam(result.lam);
      if (result.lam_locked) setCustomLam(false);
    } catch (e) {
      setTrace((previous) => ({
        ...previous, status: "error", mode,
        error: e instanceof Error ? e.message : "Request failed",
      }));
      setMessages((previous) => [
        ...previous.slice(0, -1),
        {
          prompt: text,
          error:
            e instanceof Error
              ? e.message
              : "Something went wrong. Please try again.",
        },
      ]);
      setPrompt(text);
    } finally {
      setBusy(false);
      inputRef.current?.focus();
    }
  }

  const codePrompt =
    prompt.trim() || messages.at(-1)?.prompt || "Explain an idea in a new way.";
  const code = `import os\nfrom lonnrouter import LonnRouter\n\nclient = LonnRouter(\n    api_key=os.environ["LONN_API_KEY"],\n    base_url=${JSON.stringify(endpoint)},${connection?.key_type === "application" ? "" : `\n    application=${JSON.stringify(connection?.application.name || draftApplication || "your-application")},`}\n)\n\nresult = client.generate(\n    ${JSON.stringify(codePrompt)},${mode === "direct" ? `\n    model=${JSON.stringify(model ? shortModel(model) : "eligible-model-id")},` : `\n    router="router1.0",\n    lam=${lam},`}\n)\n\nprint(result.content)`;
  const routingLabel = mode === "router" ? "router1.0" : shortModel(model);

  return (
    <div className="customer">
      <div className="customer-window">
        <header className="customer-titlebar">
          <div className="customer-masthead-routing">
            <a href="/" aria-label="Lönn workspace"><LonnLogo /></a>
            <span>ROUTING INFRASTRUCTURE</span>
          </div>
          <div className="customer-masthead-app">
            <div><MessageSquare size={28} strokeWidth={1.7} /><strong>Nordic Notes</strong></div>
            <span>EXAMPLE CUSTOMER APPLICATION</span>
          </div>
          <a
            className="customer-workspace-link"
            href="/"
            target="_blank"
            rel="noreferrer"
          >
            Lönn workspace <ArrowUpRight size={14} />
          </a>
        </header>
        <div className={`customer-body ${showCode ? "" : "code-hidden"}`}>
          <div className="customer-left" hidden={!showCode}>
          <aside
            className="customer-editor"
            aria-label="Python integration code"
            hidden={!showCode}
          >
            <div className="customer-editor-top">
              <LonnLogo inverse />
              <button
                className="customer-icon-button"
                onClick={() => setShowCode(false)}
                aria-label="Hide code panel"
              >
                <X size={15} />
              </button>
            </div>
            <div className="customer-infrastructure-heading">
              <span>ROUTING INFRASTRUCTURE</span>
              <strong><Code2 size={15} /> API integration</strong>
              <small>Python code connecting your app to Lönn</small>
            </div>
            <div className="customer-file-tab">
              <span>
                <FileCode2 size={15} /> integration.py <i />
              </span>
              <span className="customer-python">Python</span>
            </div>
            <div className="customer-code-tools">
              <span>Just a few lines. Any model.</span>
              <CopyButton value={code} />
            </div>
            <pre
              className="customer-code"
              aria-label="Python integration snippet"
            >
              <code>
                {code.split("\n").map((line, index) => (
                  <span className="customer-code-line" key={index}>
                    <span className="customer-line-number" aria-hidden="true">
                      {index + 1}
                    </span>
                    <span>
                      <PythonCode code={line} />
                      {!line && " "}
                    </span>
                  </span>
                ))}
              </code>
            </pre>
            <div className="customer-editor-note">
              <span className="customer-note-icon">
                <Link2 size={17} />
              </span>
              <div>
                <strong>Your app. Lönn under the hood.</strong>
                <p>
                  This file connects your product to Lönn. Routing and policy
                  checks happen on the server.
                </p>
                <small>Keys stay in environment variables in Python.</small>
              </div>
            </div>
            <div className="customer-editor-status">
              <span>
                <i />
                Local API
              </span>
              <span>
                UTF-8 <span>Python 3</span>
              </span>
            </div>
          </aside>
          <RuntimeRouting trace={trace} />
          </div>
          <main className="customer-chat">
            <div className="customer-chat-top">
              <div className="customer-chat-title">
                {!showCode && (
                  <button
                    className="customer-icon-button"
                    onClick={() => setShowCode(true)}
                    aria-label="Show code panel"
                  >
                    <Code2 size={18} />
                  </button>
                )}
                <span>
                  <MessageSquare size={19} /> Nordic Notes
                </span>
              </div>
              <div className="customer-connection" ref={connectionRef}>
                <button
                  ref={connectionButtonRef}
                  className={`customer-connect-button ${connection ? "is-connected" : ""}`}
                  disabled={busy || connecting}
                  onClick={() => {
                    setDraftKey(key);
                    setDraftApplication(application);
                    setMenu(menu === "connection" ? null : "connection");
                  }}
                  aria-expanded={menu === "connection"}
                  aria-controls="customer-connection-panel"
                >
                  <i />
                  {connecting
                    ? "Connecting…"
                    : connection
                      ? connection.application.name
                      : "Connect Lönn"}
                  <ChevronDown size={13} />
                </button>
                {menu === "connection" && (
                  <div
                    className="customer-popover customer-connection-popover"
                    id="customer-connection-panel"
                    role="dialog"
                    aria-label="Lönn connection"
                  >
                    <div className="customer-popover-heading">
                      <strong>Connect to Lönn</strong>
                      <button
                        className="customer-icon-button"
                        onClick={closeMenu}
                        aria-label="Close connection settings"
                      >
                        <X size={15} />
                      </button>
                    </div>
                    <p>
                      Use a personal or application API key from your workspace.
                    </p>
                    <form
                      onSubmit={(e) => {
                        e.preventDefault();
                        void connect(draftKey.trim(), draftApplication.trim());
                      }}
                    >
                      <label>
                        Lönn API key
                        <input
                          autoFocus
                          type="password"
                          autoComplete="off"
                          value={draftKey}
                          onChange={(e) => setDraftKey(e.target.value)}
                          placeholder="lonn_…"
                          required
                        />
                      </label>
                      <label>
                        Application
                        <input
                          value={draftApplication}
                          onChange={(e) => setDraftApplication(e.target.value)}
                          placeholder="Required for personal keys"
                        />
                      </label>
                      <small>
                        Application keys already identify the application.
                        Credentials stay in this page’s memory.
                      </small>
                      {connectionError && (
                        <p className="customer-inline-error" role="alert">
                          {connectionError}
                        </p>
                      )}
                      <Button disabled={connecting || !draftKey.trim()}>
                        {connecting ? "Connecting…" : "Connect"}
                        <ArrowRight size={14} />
                      </Button>
                    </form>
                  </div>
                )}
              </div>
            </div>
            <div
              className={`customer-chat-scroll ${messages.length ? "has-messages" : ""}`}
            >
              {!messages.length ? (
                <div className="customer-empty">
                  <div className="customer-assistant-mark">
                    <MessageSquare size={28} strokeWidth={1.4} />
                  </div>
                  <p className="customer-kicker">
                    POWERED BY LÖNN AI
                  </p>
                  <h1>Where should we begin?</h1>
                  <p>
                    This sample application uses Lönn to route AI requests automatically.
                  </p>
                  <div className="customer-suggestions">
                    {[
                      "Explain something simply",
                      "Help me write a function",
                      "Find a fresh perspective",
                    ].map((text) => (
                      <button
                        key={text}
                        onClick={() => {
                          setPrompt(text);
                          inputRef.current?.focus();
                        }}
                      >
                        {text}
                        <ArrowUpRight size={14} />
                      </button>
                    ))}
                  </div>
                  {!connection && !connecting && (
                    <button
                      className="customer-start-connect"
                      onClick={() => setMenu("connection")}
                    >
                      Connect your Lönn account <ArrowRight size={14} />
                    </button>
                  )}
                </div>
              ) : (
                <div
                  className="customer-messages"
                  role="log"
                  aria-label="Conversation"
                  aria-live="polite"
                >
                  {messages.map((message, index) => (
                    <article className="customer-message" key={index}>
                      <div className="customer-user-message">
                        {message.prompt}
                      </div>
                      <div className="customer-assistant-message">
                        <span className="customer-message-icon">
                          <Sparkles size={18} />
                        </span>
                        <div>
                          {message.error ? (
                            <div
                              role="alert"
                              className="customer-message-error"
                            >
                              <strong>
                                We couldn’t complete this request.
                              </strong>
                              <p>{message.error}</p>
                            </div>
                          ) : message.content !== undefined ? (
                            <>
                              <div className="customer-answer">
                                {message.content}
                              </div>
                              <small>
                                {shortModel(message.model || null)} · Total{" "}
                                {money(message.cost ?? 0)}
                                {message.billing && (
                                  <>
                                    {" "}
                                    · Provider{" "}
                                    {money(message.billing.model_cost_usd)} +
                                    Lönn{" "}
                                    {money(message.billing.service_fee_usd)}
                                  </>
                                )}{" "}
                                {message.mode === "router" && (
                                  <>
                                    · Effective λ {message.lam}
                                    {message.lambdaSource &&
                                      ` · ${message.lambdaLocked ? "enforced by " : ""}${message.lambdaSource}`}
                                  </>
                                )}
                                {message.finish === "length"
                                  ? " · Output limit reached"
                                  : ""}
                              </small>
                            </>
                          ) : (
                            <span className="customer-thinking">
                              <i />
                              <i />
                              <i />
                              <span>Finding the right model…</span>
                            </span>
                          )}
                        </div>
                      </div>
                    </article>
                  ))}
                  <div ref={bottomRef} />
                </div>
              )}
            </div>
            <div className="customer-compose-area">
              {connectionError && menu !== "connection" && (
                <p className="customer-inline-error" role="alert">
                  {connectionError}{" "}
                  <button onClick={() => setMenu("connection")}>
                    Check connection
                  </button>
                </p>
              )}
              {connection && !models.length && (
                <p className="customer-inline-error" role="status">
                  No eligible models. Review your workspace’s model pool and
                  policies.
                </p>
              )}
              <form className="customer-composer" onSubmit={send}>
                <textarea
                  ref={inputRef}
                  aria-label="Message"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  onKeyDown={(e) => {
                    if (
                      e.key === "Enter" &&
                      !e.shiftKey &&
                      !e.nativeEvent.isComposing
                    ) {
                      e.preventDefault();
                      void send(e);
                    }
                  }}
                  placeholder="Ask anything, or start with an idea…"
                  rows={2}
                />
                <div className="customer-composer-tools">
                  <div className="customer-routing" ref={routingRef}>
                    <button
                      ref={routingButtonRef}
                      type="button"
                      className="customer-plus"
                      aria-label="Routing options"
                      aria-expanded={menu === "routing"}
                      aria-controls="customer-routing-panel"
                      disabled={connecting}
                      onClick={() => {
                        setMenu(menu === "routing" ? null : "routing");
                        if (menu !== "routing" && connection)
                          void connect(key, application, true);
                      }}
                    >
                      <Plus size={19} />
                    </button>
                    {menu === "routing" && (
                      <div
                        className="customer-popover customer-routing-popover"
                        id="customer-routing-panel"
                        role="dialog"
                        aria-label="Routing options"
                      >
                        <div className="customer-popover-heading">
                          <strong>Make it your own</strong>
                          <button
                            type="button"
                            className="customer-icon-button"
                            aria-label="Close routing options"
                            onClick={closeMenu}
                          >
                            <X size={15} />
                          </button>
                        </div>
                        <p>One integration. The right model for the moment.</p>
                        <div className="customer-routing-modes">
                          <button
                            autoFocus
                            type="button"
                            className={mode === "router" ? "active" : ""}
                            disabled={busy || connecting}
                            onClick={() => setMode("router")}
                          >
                            <Sparkles size={14} /> router1.0
                          </button>
                          <button
                            type="button"
                            className={mode === "direct" ? "active" : ""}
                            disabled={busy || connecting}
                            onClick={() => setMode("direct")}
                          >
                            <Code2 size={14} /> Direct model
                          </button>
                        </div>
                        {mode === "router" ? (
                          <div className="customer-lambda">
                            <div>
                              <strong>Routing preference</strong>
                              <span>λ {lam.toFixed(2)}</span>
                            </div>
                            <LambdaSlider
                              id="customer-lambda"
                              value={lam}
                              disabled={
                                busy ||
                                connecting ||
                                !connection ||
                                connection.locked
                              }
                              onChange={chooseLambda}
                            />
                            <div className="customer-lambda-scale">
                              <span>0 · Cost only</span>
                              <span>{MAX_LAMBDA} · Quality only</span>
                            </div>
                            <button
                              type="button"
                              className="balance-preset"
                              disabled={
                                busy ||
                                connecting ||
                                !connection ||
                                connection.locked
                              }
                              onClick={() => chooseLambda(DEFAULT_LAMBDA)}
                            >
                              {DEFAULT_LAMBDA} · Balanced default
                            </button>
                            <small>
                              {connection?.locked ? (
                                <>
                                  <LockKeyhole size={12} /> Enforced by{" "}
                                  {connection.source}
                                </>
                              ) : (
                                <>
                                  {customLam
                                    ? "Your request preference"
                                    : `Inherited from ${connection?.source || "default"}`}{" "}
                                  · default λ {DEFAULT_LAMBDA}
                                </>
                              )}
                            </small>
                          </div>
                        ) : (
                          <label>
                            Eligible model
                            <select
                              aria-label="Direct model"
                              value={model}
                              disabled={busy || connecting || !models.length}
                              onChange={(e) => setModel(e.target.value)}
                            >
                              {!models.length && (
                                <option value="">Connect to list models</option>
                              )}
                              {models.map((m) => (
                                <option key={m.id} value={m.id}>
                                  {shortModel(m.id)}
                                </option>
                              ))}
                            </select>
                          </label>
                        )}
                        <details className="customer-eligible">
                          <summary>
                            {models.length} eligible models
                            <ChevronDown size={13} />
                          </summary>
                          <div>
                            {models.map((m) => (
                              <button
                                type="button"
                                key={m.id}
                                disabled={busy || connecting}
                                onClick={() => {
                                  setMode("direct");
                                  setModel(m.id);
                                }}
                              >
                                <span>{shortModel(m.id)}</span>
                                {mode === "direct" && model === m.id ? (
                                  <Check size={13} />
                                ) : (
                                  <ArrowUpRight size={13} />
                                )}
                              </button>
                            ))}
                          </div>
                          {!models.length && (
                            <small>
                              Connect an API key to see models allowed by your
                              policies.
                            </small>
                          )}
                        </details>
                      </div>
                    )}
                  </div>
                  <span className="customer-routing-label">
                    <Sparkles size={12} />
                    {routingLabel}
                  </span>
                  {mode === "router" && connection?.locked && (
                    <span
                      className="customer-locked-hint"
                      title={`λ ${connection.lam} enforced by ${connection.source}`}
                    >
                      <LockKeyhole size={12} />
                    </span>
                  )}
                  <button
                    className="customer-send"
                    aria-label="Send message"
                    disabled={
                      busy ||
                      connecting ||
                      !connection ||
                      !prompt.trim() ||
                      !models.length ||
                      (mode === "direct" && !model)
                    }
                  >
                    {busy ? (
                      <RefreshCw size={17} className="customer-spinner" />
                    ) : (
                      <ArrowUp size={19} />
                    )}
                  </button>
                </div>
              </form>
              <div className="customer-chat-footnote">
                <span>
                  <MessageSquare size={11} /> Powered by Lönn API
                </span>
                <span>Each message is a separate request.</span>
              </div>
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}
