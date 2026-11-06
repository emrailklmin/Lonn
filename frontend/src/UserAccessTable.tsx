import { Fragment, useEffect, useState } from "react";
import { Button } from "./components/ui/button";
import type { Hierarchy } from "./HierarchyView";
import type { DemoUser, Team } from "./types";

type Memberships = {
  team_ids: string[];
  project_ids: string[];
  application_ids: string[];
  team_admin_ids: string[];
  application_admin_ids: string[];
};

type Props = {
  data: Hierarchy;
  teams: Team[];
  users: DemoUser[];
  save: (userId: string, memberships: Memberships) => Promise<void>;
};

function memberships(data: Hierarchy, userId: string): Memberships {
  const teams = data.team_memberships.filter((m) => m.user_id === userId);
  const apps = data.application_memberships.filter((m) => m.user_id === userId);
  return {
    team_ids: teams.map((m) => m.team_id!),
    project_ids: data.project_memberships.filter((m) => m.user_id === userId).map((m) => m.project_id!),
    application_ids: apps.map((m) => m.application_id!),
    team_admin_ids: teams.filter((m) => m.role === "team_admin").map((m) => m.team_id!),
    application_admin_ids: apps.filter((m) => m.role === "application_admin").map((m) => m.application_id!),
  };
}

function MultiSelect({ label, options, selected, onChange }: {
  label: string;
  options: { id: string; name: string }[];
  selected: string[];
  onChange: (ids: string[]) => void;
}) {
  return (
    <details className="access-multiselect">
      <summary>{label} <span>{selected.length}</span></summary>
      <div className="access-options">
        {options.length ? options.map((item) => (
          <label className="check" key={item.id}>
            <input type="checkbox" checked={selected.includes(item.id)} onChange={() =>
              onChange(selected.includes(item.id) ? selected.filter((id) => id !== item.id) : [...selected, item.id])} />
            {item.name}
          </label>
        )) : <small>No options available.</small>}
      </div>
    </details>
  );
}

function AccessRow({ user, data, teams, save }: Props & { user: DemoUser }) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState(() => memberships(data, user.id));
  const [busy, setBusy] = useState(false);
  useEffect(() => setDraft(memberships(data, user.id)), [data, user.id]);
  const current = memberships(data, user.id);
  const names = (ids: string[], options: { id: string; name: string }[]) =>
    ids.map((id) => options.find((o) => o.id === id)?.name).filter(Boolean).join(", ") || "—";
  const admin = user.role === "enterprise_admin";
  const roles = [
    ...current.team_admin_ids.map((id) => `Team admin: ${teams.find((t) => t.id === id)?.name ?? id}`),
    ...current.application_admin_ids.map((id) => `Application admin: ${data.applications.find((a) => a.id === id)?.name ?? id}`),
  ];
  const projectOptions = data.projects.map((p) => ({ id: p.id, name: `${p.name} · ${teams.find((t) => t.id === p.team_id)?.name ?? ""}` }));
  const appOptions = data.applications.map((a) => ({ id: a.id, name: `${a.name} · ${data.projects.find((p) => p.id === a.project_id)?.name ?? ""}` }));

  function changeTeams(ids: string[]) {
    const projects = draft.project_ids.filter((id) => ids.includes(data.projects.find((p) => p.id === id)?.team_id ?? ""));
    const apps = draft.application_ids.filter((id) => projects.includes(data.applications.find((a) => a.id === id)?.project_id ?? ""));
    setDraft({ ...draft, team_ids: ids, project_ids: projects, application_ids: apps,
      team_admin_ids: draft.team_admin_ids.filter((id) => ids.includes(id)),
      application_admin_ids: draft.application_admin_ids.filter((id) => apps.includes(id)) });
  }
  function changeProjects(ids: string[]) {
    const parentTeams = data.projects.filter((p) => ids.includes(p.id)).map((p) => p.team_id);
    const apps = draft.application_ids.filter((id) => ids.includes(data.applications.find((a) => a.id === id)?.project_id ?? ""));
    setDraft({ ...draft, team_ids: [...new Set([...draft.team_ids, ...parentTeams])], project_ids: ids,
      application_ids: apps, application_admin_ids: draft.application_admin_ids.filter((id) => apps.includes(id)) });
  }
  function changeApps(ids: string[]) {
    const projects = data.applications.filter((a) => ids.includes(a.id)).map((a) => a.project_id);
    const parents = data.projects.filter((p) => projects.includes(p.id)).map((p) => p.team_id);
    setDraft({ ...draft, team_ids: [...new Set([...draft.team_ids, ...parents])],
      project_ids: [...new Set([...draft.project_ids, ...projects])], application_ids: ids,
      application_admin_ids: draft.application_admin_ids.filter((id) => ids.includes(id)) });
  }

  return (
    <Fragment>
      <tr>
        <td><strong>{user.name}</strong><small>{user.email}</small></td>
        <td>{admin ? "All teams" : names(current.team_ids, teams)}</td>
        <td>{admin ? "All projects" : names(current.project_ids, data.projects)}</td>
        <td>{admin ? "All applications" : names(current.application_ids, data.applications)}</td>
        <td>{admin ? "Organization admin" : roles.join(", ") || "Member"}{!user.enabled && " · Disabled"}</td>
        <td>{!admin && user.role === "enterprise_user" && <button className="text-button" aria-expanded={open} onClick={() => setOpen(!open)}>{open ? "Close" : "Edit access"}</button>}</td>
      </tr>
      {open && <tr className="access-edit-row"><td colSpan={6}>
        <div className="access-editor">
          <MultiSelect label="Teams" options={teams} selected={draft.team_ids} onChange={changeTeams} />
          <MultiSelect label="Projects" options={projectOptions} selected={draft.project_ids} onChange={changeProjects} />
          <MultiSelect label="Applications" options={appOptions} selected={draft.application_ids} onChange={changeApps} />
          <MultiSelect label="Team admin roles" options={teams.filter((t) => draft.team_ids.includes(t.id))} selected={draft.team_admin_ids} onChange={(ids) => setDraft({ ...draft, team_admin_ids: ids })} />
          <MultiSelect label="Application admin roles" options={appOptions.filter((a) => draft.application_ids.includes(a.id))} selected={draft.application_admin_ids} onChange={(ids) => setDraft({ ...draft, application_admin_ids: ids })} />
          <Button disabled={busy} onClick={async () => { setBusy(true); try { await save(user.id, draft); } finally { setBusy(false); } }}>Save access</Button>
        </div>
        <small>Selecting a project or application also adds its parent access.</small>
      </td></tr>}
    </Fragment>
  );
}

export function UserAccessTable(props: Props) {
  return (
    <article className="panel access-panel">
      <div className="panel-heading"><h2>Users & access</h2><span>{props.users.length} users</span></div>
      <div className="table-scroll"><table className="access-table">
        <thead><tr><th>User</th><th>Teams</th><th>Projects</th><th>Applications</th><th>Roles</th><th /></tr></thead>
        <tbody>{props.users.map((user) => <AccessRow key={user.id} user={user} {...props} />)}</tbody>
      </table></div>
    </article>
  );
}
