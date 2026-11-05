import { useState } from "react";
import { Button } from "./components/ui/button";
import { ModelRadar } from "./components/ModelRadar";
import { providerCost, lonnFee } from "./spending";
import {
  api,
  shortModel,
  money,
  type Team,
  type Catalog,
  type Usage,
} from "./types";
export function TeamsView({
  teams,
  catalog,
  usage,
  refresh,
}: {
  teams: Team[];
  catalog: Catalog[];
  usage: Usage[];
  refresh: () => Promise<void>;
}) {
  const [name, setName] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  return (
    <>
      <div className="notice">
        <div>
          <strong>Organization rules always apply</strong>
          <p>
            Each team has model restrictions; application keys provide usage
            attribution. Team controls can narrow the organization pool; they
            cannot bypass organization privacy or model exclusions.
          </p>
        </div>
      </div>
      <form
        className="panel team-create"
        onSubmit={async (e) => {
          e.preventDefault();
          setBusy(true);
          setError("");
          try {
            await api("/api/teams", {
              method: "POST",
              body: JSON.stringify({ name }),
            });
            setName("");
            await refresh();
          } catch (e) {
            setError(String(e));
          } finally {
            setBusy(false);
          }
        }}
      >
        <label>
          New team name
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Product Research"
            required
            maxLength={80}
          />
        </label>
        <Button disabled={busy || !name.trim()}>Create team</Button>
      </form>
      {error && (
        <div className="error" role="alert">
          {error}
        </div>
      )}
      {teams.map((t) => (
        <TeamCard
          key={t.id}
          team={t}
          catalog={catalog}
          usage={usage.filter((r) => r.team_id === t.id)}
          refresh={refresh}
        />
      ))}
    </>
  );
}
function TeamCard({
  team,
  catalog,
  usage,
  refresh,
}: {
  team: Team;
  catalog: Catalog[];
  usage: Usage[];
  refresh: () => Promise<void>;
}) {
  const [draft, setDraft] = useState(team),
    [editing, setEditing] = useState(false),
    [message, setMessage] = useState(""),
    [busy, setBusy] = useState(false);
  async function action(path: string, options: RequestInit, text: string) {
    setBusy(true);
    setMessage("");
    try {
      await api(path, options);
      await refresh();
      setMessage(text);
      setEditing(false);
    } catch (e) {
      setMessage(String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <article className="panel">
      <div className="panel-heading">
        <div>
          <h2>{team.name}</h2>
          <small>
            {team.eligible_model_ids.length} eligible models ·{" "}
            {team.enabled ? "Team active" : "Team access revoked"}
          </small>
        </div>
        <span className={team.enabled ? "status" : "badge"}>
          {team.enabled ? "Active" : "Revoked"}
        </span>
      </div>
      <div className="model-stats">
        <span>
          Requests<strong>{usage.length}</strong>
        </span>
        <span>
          Tokens
          <strong>
            {usage
              .reduce((s, r) => s + r.input_tokens + r.output_tokens, 0)
              .toLocaleString()}
          </strong>
        </span>
        <span>
          Total customer spend
          <strong>{money(usage.reduce((s, r) => s + r.cost_usd, 0))}</strong>
        </span>
        <span>
          Provider cost
          <strong>
            {usage.every((r) => providerCost(r) !== null)
              ? money(usage.reduce((s, r) => s + providerCost(r)!, 0))
              : "Unavailable"}
          </strong>
        </span>
        <span>
          Lönn fees
          <strong>
            {usage.every((r) => lonnFee(r) !== null)
              ? money(usage.reduce((s, r) => s + lonnFee(r)!, 0))
              : "Unavailable"}
          </strong>
        </span>
      </div>
      <div className="actions">
        <Button
          variant="outline"
          onClick={() => {
            setDraft(team);
            setEditing(!editing);
          }}
        >
          {editing ? "Cancel editing" : "Manage team policy"}
        </Button>
        <Button
          variant="ghost"
          disabled={busy}
          onClick={() =>
            void action(
              `/api/teams/${team.id}`,
              {
                method: "PUT",
                body: JSON.stringify({
                  name: team.name,
                  enabled: !team.enabled,
                  disabled_models: team.disabled_models,
                }),
              },
              team.enabled ? "Team access revoked." : "Team access restored.",
            )
          }
        >
          {team.enabled ? "Revoke access" : "Restore access"}
        </Button>
      </div>
      {editing && (
        <div className="team-policy">
          <label>
            Team name
            <input
              value={draft.name}
              onChange={(e) => setDraft({ ...draft, name: e.target.value })}
            />
          </label>
          <h3>Allowed models for this team</h3>
          {catalog
            .filter((m) => m.active)
            .map((m) => (
              <label className="check" key={m.id}>
                <input
                  type="checkbox"
                  checked={!draft.disabled_models.includes(m.id)}
                  onChange={(e) =>
                    setDraft({
                      ...draft,
                      disabled_models: e.target.checked
                        ? draft.disabled_models.filter((id) => id !== m.id)
                        : [...draft.disabled_models, m.id],
                    })
                  }
                />
                <span>
                  {shortModel(m.id)}
                  <small>
                    {m.eligible
                      ? "Permitted by organization"
                      : `Organization exclusion: ${m.exclusion_reasons.join("; ")}`}
                  </small>
                </span>
              </label>
            ))}
          <ModelRadar
            models={[]}
            coverage={catalog.filter(
              (m) => m.eligible && !draft.disabled_models.includes(m.id),
            )}
          />
          <Button
            disabled={busy || !draft.name.trim()}
            onClick={() =>
              void action(
                `/api/teams/${team.id}`,
                {
                  method: "PUT",
                  body: JSON.stringify({
                    name: draft.name,
                    enabled: draft.enabled,
                    disabled_models: draft.disabled_models,
                  }),
                },
                "Team policy saved.",
              )
            }
          >
            Save team policy
          </Button>
        </div>
      )}
      {message && <p role="status">{message}</p>}
    </article>
  );
}
