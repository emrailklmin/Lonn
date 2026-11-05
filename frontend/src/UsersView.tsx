import { useEffect, useState } from "react";
import { Button } from "./components/ui/button";
import { api, type Team, type DemoUser } from "./types";
export function UsersView({ teams }: { teams: Team[] }) {
  const [users, setUsers] = useState<DemoUser[]>([]),
    [name, setName] = useState(""),
    [email, setEmail] = useState(""),
    [password, setPassword] = useState(""),
    [team, setTeam] = useState(teams[0]?.id ?? ""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  async function refresh() {
    setUsers(await api<DemoUser[]>("/api/users"));
  }
  useEffect(() => {
    void refresh().catch((e) => setError(String(e)));
  }, []);
  async function update(user: DemoUser, values: Partial<DemoUser>) {
    setBusy(true);
    setError("");
    try {
      await api(`/api/users/${user.id}`, {
        method: "PUT",
        body: JSON.stringify({
          team_id: user.team_id,
          enabled: user.enabled,
          ...values,
        }),
      });
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <div className="notice">
        <div>
          <strong>People in your organization</strong>
          <p>
            Administrators control team membership and login access. Team
            members see only their assigned team’s usage and enforced model
            pool. Personal accounts remain separate.
          </p>
        </div>
      </div>
      {error && (
        <div className="error" role="alert">
          {error}
        </div>
      )}
      <article className="panel">
        <h2>Organization users</h2>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Name / login</th>
                <th>Role</th>
                <th>Team</th>
                <th>Access</th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.id}>
                  <td>
                    {u.name}
                    <small>{u.email}</small>
                  </td>
                  <td>
                    {u.role === "enterprise_admin"
                      ? "Administrator"
                      : "Team member"}
                  </td>
                  <td>
                    {u.role === "enterprise_admin" ? (
                      "All teams"
                    ) : (
                      <select
                        aria-label={`Team for ${u.name}`}
                        disabled={busy}
                        value={u.team_id ?? ""}
                        onChange={(e) =>
                          void update(u, { team_id: e.target.value })
                        }
                      >
                        {teams.map((t) => (
                          <option key={t.id} value={t.id}>
                            {t.name}
                          </option>
                        ))}
                      </select>
                    )}
                  </td>
                  <td>
                    {u.role === "enterprise_admin" ? (
                      <span className="status">Active</span>
                    ) : (
                      <Button
                        disabled={busy}
                        variant="outline"
                        onClick={() => void update(u, { enabled: !u.enabled })}
                      >
                        {u.enabled ? "Disable login" : "Enable login"}
                      </Button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </article>
      <article className="panel">
        <h2>Create a team member login</h2>
        <form
          onSubmit={async (e) => {
            e.preventDefault();
            setBusy(true);
            setError("");
            try {
              await api("/api/users", {
                method: "POST",
                body: JSON.stringify({ name, email, password, team_id: team }),
              });
              setName("");
              setEmail("");
              setPassword("");
              await refresh();
            } catch (e) {
              setError(String(e));
            } finally {
              setBusy(false);
            }
          }}
        >
          <label>
            Name
            <input
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              maxLength={80}
            />
          </label>
          <label>
            Login email
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </label>
          <label>
            Initial demo password
            <input
              type="password"
              required
              minLength={10}
              autoComplete="new-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </label>
          <label>
            Team
            <select
              required
              value={team}
              onChange={(e) => setTeam(e.target.value)}
            >
              {teams.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name}
                </option>
              ))}
            </select>
          </label>
          <Button disabled={busy || !team}>Create user</Button>
        </form>
        <small>
          Passwords are hashed in the local database. This demo does not send
          invitations or provide password recovery.
        </small>
      </article>
    </>
  );
}
