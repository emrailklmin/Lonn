import { Activity, Check, LoaderCircle } from "lucide-react";
import { attributes } from "../modelAptitudes";
import { shortModel } from "../types";

export type RuntimeTrace = {
  status: "pending" | "generating" | "complete" | "error";
  mode: "router" | "direct";
  lam?: number;
  topics?: Record<string, number>;
  eligible?: string[];
  ranking?: { model_id: string; score: number }[];
  selected?: string;
  attempts?: { model_id: string; check: string }[];
  attemptedModels?: string[];
  escalated?: boolean;
  error?: string;
};

const topicName = (id: string) => attributes.find((item) => item.key === id)?.label ?? id.replaceAll("_", " ");

export function RuntimeRouting({ trace }: { trace: RuntimeTrace | null }) {
  const topics = Object.entries(trace?.topics ?? {})
    .filter(([, value]) => value > 0)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);
  const ranking = trace?.ranking?.slice(0, 3) ?? [];

  return (
    <section className="runtime-panel" aria-label="Runtime routing" aria-live="polite">
      <div className="runtime-heading">
        <span><Activity size={14} /> LIVE ROUTING ENGINE</span>
        <span className={`runtime-status ${trace?.status === "pending" || trace?.status === "generating" ? "is-active" : ""}`}>
          <i />{!trace ? "READY" : trace.status === "pending" ? "ANALYZING" : trace.status === "generating" ? "GENERATING" : trace.status === "error" ? "ERROR" : "COMPLETE"}
        </span>
      </div>
      {!trace ? (
        <div className="runtime-empty">Send a message to see how it is routed.</div>
      ) : (
        <div className="runtime-flow">
          <div className="runtime-step">
            <span className="runtime-marker"><Check size={10} /></span>
            <div><strong>Prompt sent</strong><small>Request sent to Lönn</small></div>
          </div>
          {trace.status === "pending" && (
            <div className="runtime-step runtime-waiting">
              <span className="runtime-marker"><LoaderCircle size={11} /></span>
              <div><strong>Task analysis</strong><small>Analyzing your request</small></div>
            </div>
          )}
          {trace.status !== "pending" && topics.length > 0 && (
            <>
              <div className="runtime-step runtime-reveal">
                <span className="runtime-marker" />
                <div><strong>Task analysis</strong>
                  <div className="runtime-topics">
                    {topics.map(([name, probability]) => (
                      <div className="runtime-topic" key={name}>
                        <span>{topicName(name)}</span><b>{Math.round(probability * 100)}%</b>
                        <i><em style={{ width: `${Math.max(0, Math.min(100, probability * 100))}%` }} /></i>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <div className="runtime-step runtime-reveal">
                <span className="runtime-marker" />
                <div><strong>Routing mode</strong>
                  <small>{trace.mode === "router" ? 'router="router1.0"' : "Direct model selection"}
                    {trace.lam !== undefined && <> · {trace.mode === "direct" ? "Policy λ" : "Effective λ"} {trace.lam.toFixed(2)}</>}
                  </small>
                </div>
              </div>
              {trace.eligible && <div className="runtime-step runtime-reveal">
                <span className="runtime-marker" />
                <div><strong>Eligible models <span className="runtime-count">{trace.eligible.length}</span></strong>
                  <div className="runtime-chips">{trace.eligible.map((model) => <span key={model}>{shortModel(model)}</span>)}</div>
                </div>
              </div>}
              {trace.selected && <div className="runtime-step runtime-reveal">
                <span className="runtime-marker" />
                <div><strong>{trace.mode === "direct" ? "Manually selected model" : trace.status === "complete" ? "Selected model" : "Routing choice"}</strong><div className={`runtime-selected ${trace.status === "generating" ? "is-generating" : ""}`}>{shortModel(trace.selected)} {trace.status === "generating" ? <LoaderCircle size={12} /> : trace.status === "complete" ? <Check size={12} /> : null}</div>
                  {trace.mode === "router" && ranking.length > 0 && <div className="runtime-ranking">
                    <span>Router ranking · score</span>
                    {ranking.map((model, i) => <div key={model.model_id}>
                      <span>{i + 1}. {shortModel(model.model_id)}</span><b>{model.score.toFixed(3)}</b>
                    </div>)}
                  </div>}
                </div>
              </div>}
              {trace.status === "generating" && <div className="runtime-step runtime-waiting">
                <span className="runtime-marker"><LoaderCircle size={11} /></span>
                <div><strong>Processing request</strong><small>Waiting for response</small></div>
              </div>}
              {trace.status === "complete" && <div className="runtime-step runtime-reveal">
                <span className="runtime-marker"><Check size={10} /></span>
                <div><strong>Response returned</strong>
                  {trace.attemptedModels?.[0] && trace.attempts?.[0] &&
                    shortModel(trace.attemptedModels[0]) !== shortModel(trace.attempts[0].model_id) &&
                    <small>Provider fallback used</small>}
                  {trace.escalated && <small>Answer check triggered an escalation</small>}
                  {(trace.attemptedModels?.length ?? 0) > 1 && <small>Calls: {trace.attemptedModels!.map(shortModel).join(" → ")}</small>}
                </div>
              </div>}
            </>
          )}
          {trace.status === "error" && (
            <div className="runtime-step runtime-error">
              <span className="runtime-marker">!</span>
              <div><strong>Request failed</strong><small>{trace.error}</small></div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
