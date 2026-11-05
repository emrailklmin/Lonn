import { useEffect, useState } from "react";
import { LambdaSlider } from "./components/LambdaSlider";
import { Button } from "./components/ui/button";
import { CopyButton } from "./components/CopyButton";
import { api, type Team, type DemoUser } from "./types";
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
type Application = {
  id: string;
  name: string;
  project_id: string;
  api_key?: string;
  enforced_lam: number | null;
  lam_locked: boolean;
  lambda_state: LambdaState;
  parent_lock: string | null;
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
  isAdmin,
  isPrivate,
}: {
  teams: Team[];
  isAdmin: boolean;
  isPrivate: boolean;
}) {
  const [data, setData] = useState<Hierarchy | null>(null),
    [users, setUsers] = useState<DemoUser[]>([]),
    [target, setTarget] = useState(""),
    [name, setName] = useState(""),
    [parent, setParent] = useState(""),
    [kind, setKind] = useState("project"),
    [error, setError] = useState("");
  async function refresh() {
    const h = await api<Hierarchy>("/api/hierarchy");
    setData(h);
    if (isAdmin) {
      const u = await api<DemoUser[]>("/api/users");
      setUsers(u);
      if (!target)
        setTarget(u.find((x) => x.role === "enterprise_user")?.id ?? "");
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
      <div className="notice">
        <div>
          <strong>Organization → teams → projects → applications</strong>
          <p>
            Personal keys require an assigned application. Application keys
            always attribute requests to their own application.
          </p>
        </div>
      </div>
      {error && (
        <div className="error" role="alert">
          {error}
        </div>
      )}
      {isAdmin && (
        <article className="panel">
          <h2>Create a project or application</h2>
          <label>
            Type
            <select
              value={kind}
              onChange={(e) => {
                setKind(e.target.value);
                setParent("");
              }}
            >
              <option value="project">Project</option>
              <option value="application">Application</option>
            </select>
          </label>
          <label>
            Name
            <input value={name} onChange={(e) => setName(e.target.value)} />
          </label>
          <label>
            {kind === "project" ? "Team" : "Project"}
            <select value={parent} onChange={(e) => setParent(e.target.value)}>
              <option value="">Select…</option>
              {(kind === "project" ? teams : projects).map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>
          <Button
            disabled={!name.trim() || !parent}
            onClick={async () => {
              await send(
                `/api/${kind === "project" ? "projects" : "applications"}`,
                "POST",
                { name, parent_id: parent },
              );
              setName("");
            }}
          >
            Create {kind}
          </Button>
        </article>
      )}
      {teams.map((team) => (
        <article className="panel" key={team.id}>
          <h2>{team.name}</h2>
          <Policy
            label="Team-enforced λ"
            value={team.enforced_lam ?? null}
            onSave={(v) =>
              send(`/api/teams/${team.id}/lambda`, "PUT", { lam: v })
            }
            canEdit={
              !data.organization_lock &&
              (isAdmin ||
                data.team_memberships.some(
                  (m) => m.team_id === team.id && m.role === "team_admin",
                ))
            }
          />
          {data.organization_lock && (
            <small>Team λ is locked by the organization.</small>
          )}
          {projects
            .filter((p) => p.team_id === team.id)
            .map((project) => (
              <section className="hierarchy-project" key={project.id}>
                <h3>{project.name}</h3>
                <ScopedPolicy
                  id={`project-${project.id}`}
                  label="Project λ"
                  value={project.configured_lam}
                  locked={project.lam_locked}
                  effective={project.lambda_state}
                  parentLock={project.parent_lock}
                  canEdit={
                    isAdmin ||
                    isPrivate ||
                    data.team_memberships.some(
                      (m) => m.team_id === team.id && m.role === "team_admin",
                    )
                  }
                  onSave={(lam, lock) =>
                    send(`/api/projects/${project.id}/lambda`, "PUT", {
                      lam,
                      lock,
                    })
                  }
                />
                {apps
                  .filter((a) => a.project_id === project.id)
                  .map((a) => (
                    <div className="hierarchy-app" key={a.id}>
                      <strong>{a.name}</strong>
                      <small>Application ID: {a.id}</small>
                      <EffectiveLambda
                        key={`${a.id}-${a.enforced_lam}-${a.lam_locked}-${project.configured_lam}-${project.lam_locked}-${team.enforced_lam}-${data.organization_lock}`}
                        applicationId={a.id}
                      />
                      <ScopedPolicy
                        id={`app-${a.id}`}
                        label="Application λ"
                        value={a.enforced_lam}
                        locked={a.lam_locked}
                        effective={a.lambda_state}
                        parentLock={a.parent_lock}
                        canEdit={
                          isAdmin ||
                          isPrivate ||
                          data.application_memberships.some(
                            (m) =>
                              m.application_id === a.id &&
                              m.role === "application_admin",
                          )
                        }
                        onSave={(lam, lock) =>
                          send(`/api/applications/${a.id}/lambda`, "PUT", {
                            lam,
                            lock,
                          })
                        }
                      />
                      {a.api_key && (
                        <ApplicationKey
                          value={a.api_key}
                          rotate={() =>
                            send(`/api/applications/${a.id}/rotate`, "POST", {})
                          }
                        />
                      )}
                    </div>
                  ))}
              </section>
            ))}
        </article>
      ))}
      <PersonalPreference applications={apps} />
      {isAdmin && (
        <MembershipEditor
          data={data}
          teams={teams}
          users={users}
          target={target}
          setTarget={setTarget}
          save={(v) => send(`/api/users/${target}/memberships`, "PUT", v)}
        />
      )}
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
          max="1"
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
    numeric <= 1;
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
              <span>0 · More accuracy weight</span>
              <span>1 · More cost weight</span>
            </div>
          </div>
          <label>
            Exact λ
            <input
              aria-label={`${label} exact value`}
              type="number"
              min="0"
              max="1"
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

function MembershipEditor({
  data,
  teams,
  users,
  target,
  setTarget,
  save,
}: {
  data: Hierarchy;
  teams: Team[];
  users: DemoUser[];
  target: string;
  setTarget: (s: string) => void;
  save: (v: unknown) => Promise<void>;
}) {
  const [teamIds, setTeams] = useState<string[]>([]),
    [projectIds, setProjects] = useState<string[]>([]),
    [appIds, setApps] = useState<string[]>([]),
    [teamAdmins, setTeamAdmins] = useState<string[]>([]),
    [appAdmins, setAppAdmins] = useState<string[]>([]);
  useEffect(() => {
    setTeams(
      data.team_memberships
        .filter((m) => m.user_id === target)
        .map((m) => m.team_id!),
    );
    setProjects(
      data.project_memberships
        .filter((m) => m.user_id === target)
        .map((m) => m.project_id!),
    );
    setApps(
      data.application_memberships
        .filter((m) => m.user_id === target)
        .map((m) => m.application_id!),
    );
    setTeamAdmins(
      data.team_memberships
        .filter((m) => m.user_id === target && m.role === "team_admin")
        .map((m) => m.team_id!),
    );
    setAppAdmins(
      data.application_memberships
        .filter((m) => m.user_id === target && m.role === "application_admin")
        .map((m) => m.application_id!),
    );
  }, [target, data]);
  const toggle = (id: string, items: string[], set: (x: string[]) => void) =>
    set(items.includes(id) ? items.filter((x) => x !== id) : [...items, id]);
  return (
    <article className="panel">
      <h2>Assign user access</h2>
      <label>
        User
        <select value={target} onChange={(e) => setTarget(e.target.value)}>
          {users
            .filter((u) => u.role === "enterprise_user")
            .map((u) => (
              <option key={u.id} value={u.id}>
                {u.name} · {u.email}
              </option>
            ))}
        </select>
      </label>
      <p>
        Choose each team, its projects, and its applications. Admin roles can
        enforce λ at their assigned scope.
      </p>
      {teams.map((t) => (
        <div key={t.id}>
          <label className="check">
            <input
              type="checkbox"
              checked={teamIds.includes(t.id)}
              onChange={() => toggle(t.id, teamIds, setTeams)}
            />
            <strong>{t.name}</strong>
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={teamAdmins.includes(t.id)}
              onChange={() => toggle(t.id, teamAdmins, setTeamAdmins)}
            />
            Team admin
          </label>
          {data.projects
            .filter((p) => p.team_id === t.id)
            .map((p) => (
              <div className="membership-indent" key={p.id}>
                <label className="check">
                  <input
                    type="checkbox"
                    checked={projectIds.includes(p.id)}
                    onChange={() => toggle(p.id, projectIds, setProjects)}
                  />
                  {p.name}
                </label>
                {data.applications
                  .filter((a) => a.project_id === p.id)
                  .map((a) => (
                    <div className="membership-indent" key={a.id}>
                      <label className="check">
                        <input
                          type="checkbox"
                          checked={appIds.includes(a.id)}
                          onChange={() => toggle(a.id, appIds, setApps)}
                        />
                        {a.name}
                      </label>
                      <label className="check">
                        <input
                          type="checkbox"
                          checked={appAdmins.includes(a.id)}
                          onChange={() => toggle(a.id, appAdmins, setAppAdmins)}
                        />
                        Application admin
                      </label>
                    </div>
                  ))}
              </div>
            ))}
        </div>
      ))}
      <Button
        disabled={!target}
        onClick={() =>
          void save({
            team_ids: teamIds,
            project_ids: projectIds,
            application_ids: appIds,
            team_admin_ids: teamAdmins,
            application_admin_ids: appAdmins,
          })
        }
      >
        Save assignments
      </Button>
    </article>
  );
}

function EffectiveLambda({ applicationId }: { applicationId: string }) {
  const [state, setState] = useState<{
    lam: number;
    source: string;
    locked: boolean;
  } | null>(null);
  useEffect(() => {
    void api<typeof state>(`/api/applications/${applicationId}/lambda`).then(
      setState,
    );
  }, [applicationId]);
  return state ? (
    <p>
      Effective λ: <strong>{state.lam.toFixed(2)}</strong> · {state.source}
      {state.locked ? " (locked)" : ""}
    </p>
  ) : null;
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
          max="1"
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
