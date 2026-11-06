import { useEffect, useState } from "react";
import "./HierarchyView.css";
import { MAX_LAMBDA, LambdaSlider } from "./components/LambdaSlider";
import { Button } from "./components/ui/button";
import { CopyButton } from "./components/CopyButton";
import { api, type Team, type DemoUser, type Catalog, type Usage } from "./types";
import { UserAccessTable } from "./UserAccessTable";
import { TeamsView } from "./TeamsView";
import { UsersView } from "./UsersView";
type LambdaState = { lam: number; source: string; locked: boolean };
type Project = {
  id: string;
  name: string;
  team_id: string;
  configured_lam: number | null;
  lam_locked: boolean;
  lambda_state: LambdaState;
  parent_lock: string | null;
};
type Security = {
  require_zdr: boolean;
  forbid_training: boolean;
  require_encryption: boolean;
  require_private_network: boolean;
  require_audit_logs: boolean;
  allowed_regions: string[] | null;
};
type Application = {
  id: string;
  name: string;
  project_id: string;
  api_key?: string;
  enforced_lam: number | null;
  lam_locked: boolean;
  lambda_state: LambdaState;
  parent_lock: string | null;
  security_overrides: Security;
  security_effective: Security;
  security_inherited?: Security;
  eligible_model_count: number;
};
type Member = {
  user_id: string;
  team_id?: string;
  project_id?: string;
  application_id?: string;
  role?: string;
};
export type Hierarchy = {
  organization_lock: boolean;
  projects: Project[];
  applications: Application[];
  team_memberships: Member[];
  project_memberships: Member[];
  application_memberships: Member[];
};
export function HierarchyView({
  teams,
  catalog,
  usage,
  refreshWorkspace,
  isAdmin,
  isPrivate,
}: {
  teams: Team[];
  catalog: Catalog[];
  usage: Usage[];
  refreshWorkspace: () => Promise<void>;
  isAdmin: boolean;
  isPrivate: boolean;
}) {
  const [data, setData] = useState<Hierarchy | null>(null),
    [users, setUsers] = useState<DemoUser[]>([]),
    [name, setName] = useState(""),
    [parent, setParent] = useState(""),
    [kind, setKind] = useState("project"),
    [error, setError] = useState(""),
    [expandedApp, setExpandedApp] = useState(""),
    [tab, setTab] = useState<"applications" | "users">("applications");
  async function refresh() {
    const h = await api<Hierarchy>("/api/hierarchy");
    setData(h);
    if (isAdmin) {
      const u = await api<DemoUser[]>("/api/users");
      setUsers(u);
    }
  }
  useEffect(() => {
    void refresh().catch((e) => setError(String(e)));
  }, []);
  async function send(path: string, method: string, body: unknown) {
    setError("");
    try {
      await api(path, { method, body: JSON.stringify(body) });
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }
  if (!data) return <p>Loading hierarchy… {error}</p>;
  const projects = data.projects,
    apps = data.applications;
  return (
    <>
      {error && <div className="error" role="alert">{error}</div>}
      {isAdmin && <div className="management-tabs" role="tablist" aria-label="Management">
        <button role="tab" aria-selected={tab === "applications"} onClick={() => setTab("applications")}>Applications</button>
        <button role="tab" aria-selected={tab === "users"} onClick={() => setTab("users")}>Users</button>
      </div>}
      {tab === "applications" ? <>
      {isAdmin && <details className="panel hierarchy-create">
        <summary>Create a project or application</summary>
        <div className="hierarchy-create-fields">
          <label>Type<select value={kind} onChange={(e) => { setKind(e.target.value); setParent(""); }}>
            <option value="project">Project</option><option value="application">Application</option>
          </select></label>
          <label>Name<input value={name} onChange={(e) => setName(e.target.value)} /></label>
          <label>{kind === "project" ? "Team" : "Project"}
            <select value={parent} onChange={(e) => setParent(e.target.value)}>
              <option value="">Select…</option>
              {(kind === "project" ? teams : projects).map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
            </select>
          </label>
          <Button disabled={!name.trim() || !parent} onClick={async () => {
            await send(`/api/${kind === "project" ? "projects" : "applications"}`, "POST", { name, parent_id: parent });
            setName("");
          }}>Create {kind}</Button>
        </div>
      </details>}
      <article className="panel hierarchy-table-panel">
        <div className="panel-heading"><div><h2>Applications</h2><small>Applications inherit their project, team and organization policies.</small></div></div>
        <div className="table-scroll">
          <table className="hierarchy-table">
            <thead><tr><th>Application</th><th>Project</th><th>Team</th><th>API key</th><th>Effective λ</th><th>Security</th><th><span className="sr-only">Details</span></th></tr></thead>
            {apps.map((app) => {
              const project = projects.find((p) => p.id === app.project_id);
              const team = teams.find((t) => t.id === project?.team_id);
              if (!project || !team) return null;
              const open = expandedApp === app.id;
              const canEditApp = isAdmin || isPrivate || data.application_memberships.some(
                (m) => m.application_id === app.id && m.role === "application_admin",
              );
              const canEditProject = isAdmin || isPrivate || data.team_memberships.some(
                (m) => m.team_id === team.id && m.role === "team_admin",
              );
              return <tbody key={app.id}>
                <tr>
                  <td><strong>{app.name}</strong></td><td>{project.name}</td><td>{team.name}</td>
                  <td>{app.api_key ? <code>••••••••</code> : <small>Restricted</small>}</td>
                  <td><strong>{app.lambda_state.lam.toFixed(2)}</strong><small className="hierarchy-sub">{app.lambda_state.source}{app.lambda_state.locked ? " · enforced" : ""}</small></td>
                  <td><strong>{app.eligible_model_count} eligible</strong><small className="hierarchy-sub">{Object.entries(app.security_overrides ?? {}).some(([key, value]) => key === "allowed_regions" ? value !== null : value === true) ? "Additional restrictions" : "Inherited"}</small></td>
                  <td><button type="button" className="hierarchy-expand" aria-expanded={open}
                    onClick={() => setExpandedApp(open ? "" : app.id)}>{open ? "Close" : "Manage"}</button></td>
                </tr>
                {open && <tr className="hierarchy-details"><td colSpan={7}>
                  <div className="hierarchy-details-grid">
                    <div>
                      <ScopedPolicy id={`app-${app.id}`} label="Application λ" value={app.enforced_lam}
                        locked={app.lam_locked} effective={app.lambda_state} parentLock={app.parent_lock}
                        canEdit={canEditApp} onSave={(lam, lock) => send(`/api/applications/${app.id}/lambda`, "PUT", { lam, lock })} />
                    </div>
                    <div className="hierarchy-detail-side">
                      <SecurityControls value={app.security_overrides} effective={app.security_effective}
                        inherited={app.security_inherited} canEdit={canEditApp}
                        onSave={(security) => send(`/api/applications/${app.id}/security`, "PUT", security)} />
                      <strong>Application key</strong>
                      {app.api_key ? <ApplicationKey value={app.api_key} rotate={() => send(`/api/applications/${app.id}/rotate`, "POST", {})} /> : <small>Available to organization administrators.</small>}
                      <details><summary>Project λ</summary>
                        <ScopedPolicy id={`project-${project.id}`} label="Project λ" value={project.configured_lam}
                          locked={project.lam_locked} effective={project.lambda_state} parentLock={project.parent_lock}
                          canEdit={canEditProject} onSave={(lam, lock) => send(`/api/projects/${project.id}/lambda`, "PUT", { lam, lock })} />
                      </details>
                    </div>
                  </div>
                </td></tr>}
              </tbody>;
            })}
          </table>
          {!apps.length && <p>No applications available.</p>}
        </div>
      </article>
      <details className="panel hierarchy-team-policies"><summary>Team λ policies</summary>
        <div className="hierarchy-team-list">{teams.map((team) => <div key={team.id}>
          <strong>{team.name}</strong>
          <Policy label="Team-enforced λ" value={team.enforced_lam ?? null}
            onSave={(v) => send(`/api/teams/${team.id}/lambda`, "PUT", { lam: v })}
            canEdit={!data.organization_lock && (isAdmin || data.team_memberships.some(
              (m) => m.team_id === team.id && m.role === "team_admin"))} />
          {data.organization_lock && <small>Team λ is locked by the organization.</small>}
        </div>)}</div>
      </details>
      <details className="management-details"><summary>Your routing preference</summary><PersonalPreference applications={apps} /></details>
      {isAdmin && <details className="management-details"><summary>Manage teams</summary>
        <TeamsView teams={teams} catalog={catalog} usage={usage} refresh={async () => { await refreshWorkspace(); await refresh(); }} />
      </details>}
      </> : <>
        <UserAccessTable data={data} teams={teams} users={users}
          save={(userId, value) => send(`/api/users/${userId}/memberships`, "PUT", value)} />
        <details className="management-details" onToggle={(event) => { if (!event.currentTarget.open) void refresh(); }}><summary>Manage user logins</summary><UsersView teams={teams} /></details>
      </>}
    </>
  );
}
function Policy({
  label,
  value,
  onSave,
  canEdit,
}: {
  label: string;
  value: number | null;
  onSave: (v: number | null) => Promise<void>;
  canEdit: boolean;
}) {
  const [val, setVal] = useState(value === null ? "" : String(value));
  useEffect(() => setVal(value === null ? "" : String(value)), [value]);
  return (
    <div>
      <label>
        {label}
        <input
          type="number"
          min="0"
          max={MAX_LAMBDA}
          step="0.01"
          value={val}
          disabled={!canEdit}
          placeholder="Inherit"
          onChange={(e) => setVal(e.target.value)}
        />
      </label>
      {canEdit && (
        <Button
          variant="outline"
          onClick={() => void onSave(val === "" ? null : Number(val))}
        >
          Save {label}
        </Button>
      )}
      <small>Blank inherits from the next policy level.</small>
    </div>
  );
}
function ScopedPolicy({
  id,
  label,
  value,
  locked,
  effective,
  parentLock,
  canEdit,
  onSave,
}: {
  id: string;
  label: string;
  value: number | null;
  locked: boolean;
  effective: LambdaState;
  parentLock: string | null;
  canEdit: boolean;
  onSave: (value: number | null, locked: boolean) => Promise<void>;
}) {
  const [mode, setMode] = useState(value === null ? "inherit" : "configured"),
    [text, setText] = useState(String(value ?? effective.lam)),
    [isLocked, setLocked] = useState(value !== null && locked);
  useEffect(() => {
    setMode(value === null ? "inherit" : "configured");
    setText(String(value ?? effective.lam));
    setLocked(value !== null && locked);
  }, [value, locked, effective.lam]);
  const numeric = Number(text);
  const valid =
    text.trim() !== "" &&
    Number.isFinite(numeric) &&
    numeric >= 0 &&
    numeric <= MAX_LAMBDA;
  const disabled = !canEdit || !!parentLock;
  const status = parentLock
    ? `Locked by ${parentLock}`
    : value === null
      ? `Inherited from ${effective.source}`
      : locked
        ? "Enforced at this level"
        : "Locally configured";
  return (
    <div className="scope-lambda">
      <div className="panel-heading">
        <strong>{label}</strong>
        <span className="badge">{status}</span>
      </div>
      <p>
        Effective λ: <strong>{effective.lam}</strong> · {effective.source}
        {effective.locked ? " (enforced)" : ""}
      </p>
      <label>
        Setting
        <select
          aria-label={`${label} mode`}
          value={mode}
          disabled={disabled}
          onChange={(e) => setMode(e.target.value)}
        >
          <option value="inherit">Inherit parent λ</option>
          <option value="configured">Configure λ</option>
        </select>
      </label>
      {mode === "configured" && (
        <>
          <div className="quality-control">
            <LambdaSlider
              id={id}
              label={label}
              value={valid ? numeric : effective.lam}
              disabled={disabled}
              onChange={(v) => setText(String(v))}
            />
            <div className="slider-scale">
              <span>0 · Cost only</span>
              <span>{MAX_LAMBDA} · Quality only</span>
            </div>
          </div>
          <label>
            Exact λ
            <input
              aria-label={`${label} exact value`}
              type="number"
              min="0"
              max={MAX_LAMBDA}
              step="any"
              value={text}
              disabled={disabled}
              onChange={(e) => setText(e.target.value)}
            />
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={isLocked}
              disabled={disabled}
              onChange={(e) => setLocked(e.target.checked)}
            />{" "}
            Lock λ for lower levels
          </label>
        </>
      )}
      {canEdit && (
        <Button
          variant="outline"
          disabled={disabled || (mode === "configured" && !valid)}
          onClick={() =>
            void onSave(
              mode === "inherit" ? null : numeric,
              mode === "configured" && isLocked,
            )
          }
        >
          Save {label}
        </Button>
      )}
      <small>
        {parentLock
          ? "A higher-level administrator must unlock this setting first."
          : "Unlocked values are inherited defaults; lower-level preferences can override them."}
      </small>
    </div>
  );
}

const securityFields = [
  ["require_zdr", "Zero data retention"],
  ["forbid_training", "No provider training"],
  ["require_encryption", "Encryption"],
  ["require_private_network", "Private network"],
  ["require_audit_logs", "Audit logs"],
] as const;
const securityRegions = [
  ["north_america", "North America"], ["south_america", "South America"],
  ["europe", "Europe"], ["asia", "Asia"], ["africa", "Africa"], ["oceania", "Oceania"],
] as const;

function SecurityControls({ value, effective, inherited, canEdit, onSave }: {
  value: Security; effective: Security; inherited?: Security; canEdit: boolean;
  onSave: (value: Security) => Promise<void>;
}) {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  if (!draft || !effective) return null;
  const inheritedRegions = inherited?.allowed_regions ?? effective.allowed_regions ?? [];
  return <details className="hierarchy-security"><summary>Security & privacy</summary>
    <small>Organization requirements always apply. Application settings can add restrictions. Model security metadata is based on demo assumptions.</small>
    {securityFields.map(([key, label]) => <label className="check" key={key}>
      <input type="checkbox" disabled={!canEdit || !!inherited?.[key]} checked={draft[key] || !!inherited?.[key]}
        onChange={(event) => setDraft({ ...draft, [key]: event.target.checked })} />
      {label} {inherited?.[key] && <small>· inherited</small>}
    </label>)}
    <label className="check"><input type="checkbox" disabled={!canEdit} checked={draft.allowed_regions !== null}
      onChange={(event) => setDraft({ ...draft, allowed_regions: event.target.checked ? [...inheritedRegions] : null })} />
      Restrict regions for this application
    </label>
    {draft.allowed_regions !== null && <div className="hierarchy-regions">{securityRegions.map(([id, label]) => {
      const allowedByParent = inherited ? !!inherited.allowed_regions?.includes(id) : true;
      return <label className="check" key={id}><input type="checkbox" disabled={!canEdit || !allowedByParent}
        checked={draft.allowed_regions!.includes(id)} onChange={(event) => setDraft({ ...draft,
          allowed_regions: event.target.checked ? [...draft.allowed_regions!, id] : draft.allowed_regions!.filter((region) => region !== id),
        })} />{label}{!allowedByParent && " · restricted by organization"}</label>;
    })}</div>}
    {canEdit && <Button variant="outline" onClick={() => void onSave(draft)}>Save security</Button>}
  </details>;
}

function PersonalPreference({ applications }: { applications: Application[] }) {
  const [id, setId] = useState(applications[0]?.id ?? ""),
    [val, setVal] = useState(""),
    [state, setState] = useState<{
      lam: number;
      source: string;
      locked: boolean;
    } | null>(null),
    [message, setMessage] = useState("");
  useEffect(() => {
    void api<{ lam: number | null }>("/api/preference").then((p) =>
      setVal(p.lam === null ? "" : String(p.lam)),
    );
  }, []);
  useEffect(() => {
    if (id)
      void api<typeof state>(`/api/applications/${id}/lambda`).then(setState);
  }, [id]);
  return (
    <article className="panel">
      <h2>Your routing preference</h2>
      <label>
        Application
        <select value={id} onChange={(e) => setId(e.target.value)}>
          {applications.map((a) => (
            <option key={a.id} value={a.id}>
              {a.name}
            </option>
          ))}
        </select>
      </label>
      {state && (
        <p>
          Effective λ: {state.lam.toFixed(2)} · {state.source}{" "}
          {state.locked ? "(locked by administrator)" : ""}
        </p>
      )}
      <label>
        Personal λ (blank uses default)
        <input
          type="number"
          min="0"
          max={MAX_LAMBDA}
          step="0.01"
          value={val}
          disabled={!!state?.locked}
          onChange={(e) => setVal(e.target.value)}
        />
      </label>
      <Button
        disabled={!id || !!state?.locked}
        onClick={async () => {
          try {
            await api("/api/preference", {
              method: "PUT",
              body: JSON.stringify({ lam: val === "" ? null : Number(val) }),
            });
            setState(await api(`/api/applications/${id}/lambda`));
            setMessage("Personal preference saved.");
          } catch (e) {
            setMessage(String(e));
          }
        }}
      >
        Save personal preference
      </Button>
      <small>{message}</small>
    </article>
  );
}

function ApplicationKey({
  value,
  rotate,
}: {
  value: string;
  rotate: () => Promise<void>;
}) {
  const [shown, setShown] = useState(false);
  return (
    <>
      <code>{shown ? value : "lonn_app_••••••••••••••••"}</code>
      <div className="actions">
        <Button variant="outline" onClick={() => setShown(!shown)}>
          {shown ? "Hide key" : "Reveal key"}
        </Button>
        <CopyButton value={value} />
        <Button variant="outline" onClick={() => void rotate()}>
          Rotate application key
        </Button>
      </div>
    </>
  );
}
