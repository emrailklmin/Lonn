import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BookOpen,
  Users,
  FolderTree,
  KeyRound,
  ArrowUpRight,
  Layers,
  LayoutDashboard,
  Leaf,
  Settings2,
  Terminal,
  RefreshCw,
} from "lucide-react";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { Button } from "./components/ui/button";
import {
  api,
  ApiError,
  shortModel,
  money,
  type Account,
  type Catalog,
  type Usage,
  type Team,
  type DemoUser,
} from "./types";
import "./style.css";
import { DeveloperGuide } from "./DeveloperGuide";
import { SpendingComparison } from "./SpendingComparison";
import { benchmarkTopCost, providerCost, lonnFee } from "./spending";
import { HierarchyView, type Hierarchy } from "./HierarchyView";
import { LoginView } from "./LoginView";
import { TeamsView } from "./TeamsView";
import { UsersView } from "./UsersView";
import { AccessView } from "./AccessView";
import { ModelPoolView } from "./ModelPoolView";
import { SettingsView } from "./SettingsView";
import { CustomerApp } from "./CustomerApp";

function Empty({ children }: { children: React.ReactNode }) {
  return <p className="empty">{children}</p>;
}
function App() {
  const [account, setAccount] = useState<Account | null>(null),
    [ready, setReady] = useState(false),
    [page, setPage] = useState("Overview");
  const [catalog, setCatalog] = useState<Catalog[]>([]),
    [usage, setUsage] = useState<Usage[]>([]),
    [error, setError] = useState("");
  const [teams, setTeams] = useState<Team[]>([]);
  const [users, setUsers] = useState<DemoUser[]>([]);
  const [hierarchy, setHierarchy] = useState<Hierarchy | null>(null);
  const [projectFilter, setProjectFilter] = useState("all"),
    [appFilter, setAppFilter] = useState("all"),
    [userFilter, setUserFilter] = useState("all");
  const [teamFilter, setTeamFilter] = useState("all");
  const [source, setSource] = useState("all"),
    [detail, setDetail] = useState<Usage | null>(null);
  async function refresh() {
    try {
      const a = await api<Account>("/api/account");
      const [c, u, t, h] = await Promise.all([
        api<Catalog[]>("/api/catalog"),
        api<Usage[]>("/api/usage"),
        api<Team[]>("/api/teams"),
        api<Hierarchy>("/api/hierarchy"),
      ]);
      setAccount(a);
      setCatalog(c);
      setUsage(u);
      setTeams(t);
      setHierarchy(h);
      setUsers(
        a.user.role === "enterprise_admin"
          ? await api<DemoUser[]>("/api/users")
          : [],
      );
      setError("");
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) setAccount(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setReady(true);
    }
  }
  useEffect(() => {
    void refresh();
  }, []);
  useEffect(() => {
    if (!account) return;
    const timer = setInterval(() => {
      void refresh();
    }, 5000);
    return () => clearInterval(timer);
  }, [!!account]);
  if (!ready) return <div className="welcome">Loading workspace…</div>;
  if (!account)
    return (
      <LoginView
        onLogin={async () => {
          setPage("Overview");
          setTeamFilter("all");
          setSource("all");
          await refresh();
        }}
      />
    );
  const isAdmin = account.user.role === "enterprise_admin";
  const isMember = account.user.role === "enterprise_user";
  const rows = usage.filter(
    (r) =>
      (source === "all" || r.source === source) &&
      (teamFilter === "all" ||
        (teamFilter === "personal" ? !r.team_id : r.team_id === teamFilter)) &&
      (projectFilter === "all" || r.project_id === projectFilter) &&
      (appFilter === "all" || r.application_id === appFilter) &&
      (userFilter === "all" ||
        (userFilter === "application-key"
          ? !r.user_id
          : r.user_id === userFilter)),
  );
  const total = rows.reduce(
    (a, r) => ({
      tokens: a.tokens + r.input_tokens + r.output_tokens,
      cost: a.cost + r.cost_usd,
      provider: a.provider + (providerCost(r) ?? 0),
      fees: a.fees + (lonnFee(r) ?? 0),
      costsKnown:
        a.costsKnown && providerCost(r) !== null && lonnFee(r) !== null,
      latency: a.latency + r.latency_ms,
      success: a.success + (r.status === "success" ? 1 : 0),
    }),
    {
      tokens: 0,
      cost: 0,
      provider: 0,
      fees: 0,
      costsKnown: true,
      latency: 0,
      success: 0,
    },
  );
  const days: Record<
    string,
    { day: string; requests: number; cost: number; tokens: number }
  > = {};
  const models: Record<
    string,
    { model: string; requests: number; cost: number; tokens: number }
  > = {};
  rows.forEach((r) => {
    const day = r.timestamp.slice(0, 10);
    const d = (days[day] ??= { day, requests: 0, cost: 0, tokens: 0 });
    d.requests++;
    d.cost += r.cost_usd;
    d.tokens += r.input_tokens + r.output_tokens;
    const name = shortModel(r.model);
    const m = (models[name] ??= {
      model: name,
      requests: 0,
      cost: 0,
      tokens: 0,
    });
    m.requests++;
    m.cost += r.cost_usd;
    m.tokens += r.input_tokens + r.output_tokens;
  });
  const timeline = Object.values(days).sort((a, b) =>
    a.day.localeCompare(b.day),
  );
  return (
    <div className="shell">
      <aside>
        <div className="brand">
          <Leaf /> lönn<span>AI</span>
        </div>
        <div className="workspace">
          <span className="avatar">
            {account.user.name
              .split(" ")
              .map((n) => n[0])
              .join("")}
          </span>
          <div>
            <strong>{account.name}</strong>
            <small>
              {isAdmin
                ? "Enterprise admin"
                : isMember
                  ? account.user.team_name
                  : "Private account"}
            </small>
          </div>
        </div>
        <p className="nav-label">WORKSPACE</p>
        <nav>
          {[
            [LayoutDashboard, "Overview"],
            [Activity, "Request history"],
            [Layers, "Model pool"],
            [Settings2, "Settings"],
            [FolderTree, "Projects & apps"],
            [BookOpen, "Developer guide"],
            ...(isAdmin
              ? [
                  [KeyRound, "Teams & access"],
                  [Users, "Users"],
                ]
              : []),
          ].map(([Icon, label]) => {
            const I = Icon as typeof Activity;
            return (
              <button
                key={String(label)}
                className={page === label ? "active" : ""}
                onClick={() => {
                  setPage(String(label));
                  setDetail(null);
                }}
              >
                <I size={18} />
                {String(label)}
              </button>
            );
          })}
        </nav>
        <div className="sidebar-bottom">
          <a href="/customer" target="_blank" rel="noreferrer">
            <Terminal size={17} /> Customer application{" "}
            <ArrowUpRight size={15} />
          </a>
          <p>
            <i /> Running locally
          </p>
          <small>Prototype 3.0 · Customer demo</small>
          <button
            onClick={async () => {
              await api("/api/demo/logout", { method: "POST" });
              setAccount(null);
            }}
          >
            Leave workspace
          </button>
        </div>
      </aside>
      <main>
        <header>
          <span>
            Workspace <span className="slash">/</span> {page}
          </span>
          <span className="badge">LOCAL DEMO</span>
        </header>
        <section className="content">
          <div className="page-heading">
            <div>
              <p className="eyebrow">
                {account.name.toUpperCase()} · {account.user.name}
              </p>
              <h1>{page}</h1>
              <p>
                {page === "Overview"
                  ? "A clear view of every request."
                  : page === "Model pool"
                    ? "The models behind your application."
                    : page === "Developer guide"
                      ? "Build with Lönn, step by step."
                      : page === "Settings"
                        ? "Shape how your intelligence works."
                        : page === "Teams & access"
                          ? "One organization. Purpose-built access for every team."
                          : page === "Users"
                            ? "The people behind your workspace."
                            : "Follow each routing decision."}
              </p>
            </div>
            <Button variant="outline" onClick={() => void refresh()}>
              <RefreshCw size={15} /> Refresh
            </Button>
          </div>
          {error && (
            <div role="alert" className="error">
              {error}
            </div>
          )}
          {(page === "Overview" || page === "Request history") && (
            <div className="toolbar">
              <label>
                Usage source{" "}
                <select
                  value={source}
                  onChange={(e) => setSource(e.target.value)}
                >
                  <option value="all">All requests</option>
                  <option value="seeded">Seeded demonstration</option>
                  <option value="live">Live requests</option>
                </select>
              </label>
              <label>
                Organization
                <select value={account.id} onChange={() => {}}>
                  <option value={account.id}>{account.name}</option>
                </select>
              </label>
              {isAdmin && (
                <label>
                  Team
                  <select
                    value={teamFilter}
                    onChange={(e) => setTeamFilter(e.target.value)}
                  >
                    <option value="all">All teams</option>
                    <option value="personal">Organization key</option>
                    {teams.map((t) => (
                      <option key={t.id} value={t.id}>
                        {t.name}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <label>
                Project
                <select
                  value={projectFilter}
                  onChange={(e) => setProjectFilter(e.target.value)}
                >
                  <option value="all">All projects</option>
                  {hierarchy?.projects.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name} · {teams.find((t) => t.id === p.team_id)?.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Application
                <select
                  value={appFilter}
                  onChange={(e) => setAppFilter(e.target.value)}
                >
                  <option value="all">All applications</option>
                  {hierarchy?.applications.map((a) => (
                    <option key={a.id} value={a.id}>
                      {a.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                User
                <select
                  value={userFilter}
                  onChange={(e) => setUserFilter(e.target.value)}
                >
                  <option value="all">All users</option>
                  <option value="application-key">
                    Application key / unassigned
                  </option>
                  {Array.from(
                    new Set(
                      usage
                        .map((r) => r.user_id)
                        .filter((id): id is string => !!id),
                    ),
                  ).map((id) => (
                    <option key={id} value={id}>
                      {users.find((u) => u.id === id)?.name ??
                        (id === account.user.id ? account.user.name : id)}
                    </option>
                  ))}
                </select>
              </label>
              <small>
                Updates every 5 seconds · Lönn fee: 20% of positive estimated
                savings
              </small>
            </div>
          )}
          {page === "Overview" && (
            <>
              <div className="metrics">
                {[
                  [
                    "Requests",
                    rows.length.toLocaleString(),
                    `${total.success} successful · ${rows.length - total.success} failed`,
                  ],
                  [
                    "Total tokens",
                    total.tokens.toLocaleString(),
                    "Input + output, across reported attempts",
                  ],
                  [
                    "Total customer spend",
                    money(total.cost),
                    total.costsKnown
                      ? `Provider ${money(total.provider)} + Lönn ${money(total.fees)} · USD`
                      : "Includes recorded fees · some cost breakdowns unavailable",
                  ],
                  [
                    "Average latency",
                    `${rows.length ? (total.latency / rows.length / 1000).toFixed(2) : "0"} s`,
                    "End-to-end server time",
                  ],
                ].map(([label, value, caption]) => (
                  <article className="metric" key={label}>
                    <p>{label}</p>
                    <strong>{value}</strong>
                    <small>{caption}</small>
                  </article>
                ))}
              </div>
              <SpendingComparison rows={rows} />
              <div className="charts">
                {(["requests", "cost", "tokens"] as const).map((key) => (
                  <article className="panel" key={key}>
                    <h2>
                      {key === "requests"
                        ? "Request volume"
                        : key === "cost"
                          ? "Total customer spending over time"
                          : "Tokens over time"}
                    </h2>
                    <div className="chart">
                      {timeline.length ? (
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={timeline}>
                            <CartesianGrid vertical={false} stroke="#e7ebe4" />
                            <XAxis
                              dataKey="day"
                              tickFormatter={(v) => v.slice(5)}
                              fontSize={11}
                            />
                            <YAxis width={55} fontSize={11} />
                            <Tooltip />
                            <Area
                              type="monotone"
                              dataKey={key}
                              stroke="#46725a"
                              fill="#e0eadd"
                              strokeWidth={2}
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      ) : (
                        <Empty>No requests in this view yet.</Empty>
                      )}
                    </div>
                  </article>
                ))}
              </div>
              <article className="panel">
                <h2>Model distribution</h2>
                <div className="chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={Object.values(models).sort(
                        (a, b) => b.requests - a.requests,
                      )}
                    >
                      <XAxis dataKey="model" fontSize={10} />
                      <YAxis allowDecimals={false} />
                      <Tooltip />
                      <Bar
                        dataKey="requests"
                        fill="#52745e"
                        radius={[4, 4, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Requests</th>
                        <th>Tokens</th>
                        <th>Customer spend</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.values(models)
                        .sort((a, b) => b.requests - a.requests)
                        .map((m) => (
                          <tr key={m.model}>
                            <td>{m.model}</td>
                            <td>{m.requests}</td>
                            <td>{m.tokens.toLocaleString()}</td>
                            <td>{money(m.cost)}</td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </article>
            </>
          )}
          {(page === "Overview" || page === "Request history") && (
            <article className="panel">
              <div className="panel-heading">
                <h2>
                  {page === "Overview" ? "Recent requests" : "Request history"}
                </h2>
                <small>Click a request for routing details</small>
              </div>
              {rows.length ? (
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Request / time</th>
                        <th>Model</th>
                        <th>Team / source</th>
                        <th>Application</th>
                        <th>User</th>
                        <th>Tokens</th>
                        <th>Customer spend</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.slice(0, page === "Overview" ? 8 : 200).map((r) => (
                        <tr key={r.id}>
                          <td>
                            <button
                              className="text-button"
                              onClick={() => setDetail(r)}
                            >
                              {r.id.slice(0, 12)}
                            </button>
                            <small>
                              {new Date(r.timestamp).toLocaleString()}
                            </small>
                          </td>
                          <td>{shortModel(r.model)}</td>
                          <td>
                            <small>
                              {teams.find((t) => t.id === r.team_id)?.name ??
                                (isAdmin ? "Organization" : "Personal")}
                            </small>
                            <span className="badge">{r.source}</span>
                          </td>
                          <td>
                            {hierarchy?.applications.find(
                              (a) => a.id === r.application_id,
                            )?.name ?? "—"}
                          </td>
                          <td>
                            {r.user_id
                              ? (users.find((u) => u.id === r.user_id)?.name ??
                                (r.user_id === account.user.id
                                  ? account.user.name
                                  : r.user_id))
                              : "Application key / historical"}
                          </td>
                          <td>{r.input_tokens + r.output_tokens}</td>
                          <td>
                            {money(r.cost_usd)}
                            <small>
                              {r.source === "seeded"
                                ? "Demo data"
                                : "Includes recorded Lönn fee"}
                            </small>
                          </td>
                          <td>
                            <span
                              className={
                                r.status === "success"
                                  ? "status"
                                  : "status failed"
                              }
                            >
                              {r.status}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <Empty>
                  No requests yet. Send a prompt from the customer application.
                </Empty>
              )}
              {detail && (
                <section className="details">
                  <div className="panel-heading">
                    <h3>Request {detail.id}</h3>
                    <Button variant="ghost" onClick={() => setDetail(null)}>
                      Close
                    </Button>
                  </div>
                  <p>
                    {detail.latency_ms.toFixed(0)} ms · Input{" "}
                    {detail.input_tokens} · Output {detail.output_tokens}
                  </p>
                  {detail.details.billing && (
                    <div className="model-stats">
                      <span>
                        Provider cost (token rates)
                        <strong>
                          {money(detail.details.billing.model_cost_usd)}
                        </strong>
                      </span>
                      <span>
                        Lönn fee (
                        {detail.details.billing.service_fee_basis ===
                        "estimated_savings_v1"
                          ? "20% of positive savings"
                          : "historical rate"}
                        )
                        <strong>
                          {money(detail.details.billing.service_fee_usd)}
                        </strong>
                      </span>
                      <span>
                        Total customer cost
                        <strong>
                          {money(detail.details.billing.total_cost_usd)}
                        </strong>
                      </span>
                    </div>
                  )}
                  <p>
                    Estimated top-model cost:{" "}
                    {benchmarkTopCost(detail) === null
                      ? "Unavailable (missing benchmark or top-model data)"
                      : money(benchmarkTopCost(detail)!)}
                  </p>
                  {benchmarkTopCost(detail) !== null &&
                    providerCost(detail) !== null && (
                      <p>
                        Estimated savings before fee:{" "}
                        {money(
                          benchmarkTopCost(detail)! - providerCost(detail)!,
                        )}{" "}
                        · after fee:{" "}
                        {money(benchmarkTopCost(detail)! - detail.cost_usd)}
                      </p>
                    )}
                  <details>
                    <summary>Full routing and billing details</summary>
                    <small>
                      Benchmark estimates and rates below are retained from the
                      routing snapshot. Reported upstream charges, when
                      available, are separate provider telemetry.
                    </small>
                    <pre>{JSON.stringify(detail.details, null, 2)}</pre>
                  </details>
                </section>
              )}
            </article>
          )}
          {page === "Developer guide" && <DeveloperGuide account={account} />}
          {page === "Projects & apps" && (
            <HierarchyView
              teams={teams}
              isAdmin={isAdmin}
              isPrivate={account.user.role === "private_user"}
            />
          )}
          {page === "Model pool" && (
            <ModelPoolView
              catalog={catalog}
              account={account}
              refresh={refresh}
            />
          )}
          {page === "Teams & access" && isAdmin && (
            <TeamsView
              teams={teams}
              catalog={catalog}
              usage={usage}
              refresh={refresh}
            />
          )}
          {page === "Users" && isAdmin && <UsersView teams={teams} />}
          {page === "Settings" && isMember && (
            <AccessView account={account} refresh={refresh} />
          )}
          {page === "Settings" && !isMember && (
            <SettingsView
              account={account}
              catalog={catalog}
              refresh={refresh}
            />
          )}
          <footer>
            Lönn AI · Local customer demonstration{" "}
            <span>
              Seeded records are synthetic. Spending comparisons are estimates.
            </span>
          </footer>
        </section>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    {location.pathname === "/customer" ? <CustomerApp /> : <App />}
  </React.StrictMode>,
);
