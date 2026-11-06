import { useState } from "react";
import { Layers, Search } from "lucide-react";
import { Button } from "./components/ui/button";
import { ModelRadar } from "./components/ModelRadar";
import { attributes } from "./modelAptitudes";
import { regionLabels } from "./components/RegionMap";
import {
  api,
  shortModel,
  money,
  type Catalog,
  type Account,
  type Settings,
} from "./types";

export function ModelPoolView({
  catalog,
  account,
  refresh,
}: {
  catalog: Catalog[];
  account: Account;
  refresh: () => Promise<void>;
}) {
  const [selected, setSelected] = useState(
    catalog
      .filter((m) => m.eligible)
      .slice(0, 3)
      .map((m) => m.id),
  );
  const [query, setQuery] = useState(""),
    [filter, setFilter] = useState("all"),
    [expanded, setExpanded] = useState<string | null>(null),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const eligible = catalog.filter((m) => m.eligible);
  const comparison = catalog.filter((m) => selected.includes(m.id));
  const visible = catalog.filter(
    (m) =>
      (filter === "all" || (filter === "eligible" && m.eligible)) &&
      m.id.toLowerCase().includes(query.toLowerCase()),
  );
  function compare(id: string) {
    setSelected(
      selected.includes(id)
        ? selected.filter((x) => x !== id)
        : selected.length < 4
          ? [...selected, id]
          : selected,
    );
  }
  async function toggle(model: Catalog) {
    setBusy(true);
    setError("");
    try {
      const latest = await api<Settings>("/api/settings");
      const ids = latest.disabled_models;
      await api("/api/settings", {
        method: "PUT",
        body: JSON.stringify({
          ...latest,
          disabled_models: ids.includes(model.id)
            ? ids.filter((id) => id !== model.id)
            : [...ids, model.id],
        }),
      });
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <div className="notice">
        <Layers size={20} />
        <div>
          <strong>
            {eligible.length} of {catalog.length} models eligible
          </strong>
          <p>
            Enable models for routing, or compare their characteristics below.
            Privacy restrictions still apply to every selection and fallback.
          </p>
          <small>
            Security and region metadata are demo assumptions; provider
            safeguards and location are not verified or controlled.
          </small>
          {!eligible.length && (
            <p role="alert">
              No models meet your current security and model pool settings.
            </p>
          )}
        </div>
      </div>
      {error && (
        <div className="error" role="alert">
          {error}
        </div>
      )}
      <article className="panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">MODEL ARENA</p>
            <h2>Different strengths. One model pool.</h2>
          </div>
          <span className="badge">CATALOG SCORES</span>
        </div>
        <p>
          Compare up to four models. The dashed outline shows the best available
          score in each category across your eligible pool, not a combined-model
          quality score.
        </p>
        <div className="comparison-picks">
          {catalog.map((m) => (
            <button
              key={m.id}
              className={
                selected.includes(m.id)
                  ? "comparison-chip selected"
                  : "comparison-chip"
              }
              aria-pressed={selected.includes(m.id)}
              disabled={!selected.includes(m.id) && selected.length >= 4}
              onClick={() => compare(m.id)}
            >
              {shortModel(m.id)}
              {selected.includes(m.id) ? " ✓" : ""}
            </button>
          ))}
        </div>
        {comparison.length || eligible.length ? (
          <ModelRadar models={comparison} coverage={eligible} />
        ) : (
          <p className="empty">
            Select a model to compare. No models are currently eligible for pool
            coverage.
          </p>
        )}
      </article>
      <div className="pool-toolbar">
        <label className="search">
          <Search size={16} />
          <input
            aria-label="Search models"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search models or providers"
          />
        </label>
        <label>
          Show
          <select value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="all">All models ({catalog.length})</option>
            <option value="eligible">
              Eligible models ({eligible.length})
            </option>
          </select>
        </label>
      </div>
      <div className="model-grid">
        {visible.map((m) => {
          const enabled = !account.settings.disabled_models.includes(m.id);
          return (
            <article className="panel model" key={m.id}>
              <div className="panel-heading">
                <div>
                  <h2>{shortModel(m.id)}</h2>
                  <small>
                    {m.provider} · {m.id.split("/")[1]}
                  </small>
                </div>
                <span className={m.eligible ? "status" : "badge"}>
                  {m.eligible ? "Eligible" : "Excluded"}
                </span>
              </div>
              <ModelRadar models={[m]} compact />
              <div className="model-stats">
                <span>
                  Base input / 1M<strong>{money(m.cost_per_m_input)}</strong>
                </span>
                <span>
                  Base output / 1M<strong>{money(m.cost_per_m_output)}</strong>
                </span>
                <span>
                  Context<strong>{m.context_window?.toLocaleString()}</strong>
                </span>
              </div>
              <small>
                Customer cost adds a Lönn fee of 20% of positive
                benchmark-estimated savings to these token charges.
              </small>
              <p className="eligibility-reason">
                {m.exclusion_reasons.join(" · ") ||
                  "Eligible for your current settings."}
              </p>
              <div className="actions">
                <Button
                  variant="outline"
                  onClick={() => setExpanded(expanded === m.id ? null : m.id)}
                >
                  {expanded === m.id ? "Hide details" : "Model details"}
                </Button>
                {m.active && account.user.role !== "enterprise_user" && (
                  <Button
                    variant={enabled ? "ghost" : "default"}
                    disabled={busy || !m.policy_eligible}
                    onClick={() => void toggle(m)}
                  >
                    {!m.policy_eligible
                      ? "Unavailable under policy"
                      : enabled
                        ? "Disable for routing"
                        : "Enable for routing"}
                  </Button>
                )}
              </div>
              {expanded === m.id && (
                <div className="model-detail">
                  <p>
                    {m.availability}. {m.privacy_status}.
                  </p>
                  <p>
                    Assumed supported regions:{" "}
                    {m.security.regions
                      .map((region) => regionLabels[region])
                      .join(", ")}
                    .
                  </p>
                  <p>
                    Allowed overlap:{" "}
                    {m.matching_regions
                      .map((region) => regionLabels[region])
                      .join(", ") || "None"}
                    .
                  </p>
                  <p>
                    Assumed safeguards:{" "}
                    {[
                      m.security.zero_data_retention && "Zero data retention",
                      m.security.no_training && "No training",
                      m.security.encryption && "Encryption",
                      m.security.private_network && "Private network",
                      m.security.audit_logs && "Audit logs",
                    ]
                      .filter(Boolean)
                      .join(", ") || "None"}
                    .
                  </p>
                  <p>Configured price source: {m.price_basis}.</p>
                  <table>
                    <thead>
                      <tr>
                        <th>Task</th>
                        <th>Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {attributes.map((a) => (
                        <tr key={a.key}>
                          <td>{a.label}</td>
                          <td>
                            {m.task_aptitudes[a.key]?.score.toFixed(4) ?? "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </article>
          );
        })}
      </div>
      {!visible.length && (
        <p className="empty">No models match these filters.</p>
      )}
    </>
  );
}
