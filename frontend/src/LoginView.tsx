import { useState } from "react";
import { ArrowUpRight } from "lucide-react";
import { Button } from "./components/ui/button";
import { LonnLogo } from "./components/LonnLogo";
import { api } from "./types";
const personas = [
  {
    name: "Enterprise administrator",
    email: "admin@nordic.demo",
    description:
      "Manage Nordic Studio, its teams, users, model policies and all organization usage.",
  },
  {
    name: "Enterprise team member",
    email: "robin@nordic.demo",
    description:
      "Use Engineering’s API key and view team usage. Your administrator controls the model pool.",
  },
  {
    name: "Private user",
    email: "sam@personal.demo",
    description:
      "Your own account, API key, model preferences and private usage history.",
  },
];
export function LoginView({ onLogin }: { onLogin: () => Promise<void> }) {
  const [email, setEmail] = useState(personas[0].email),
    [password, setPassword] = useState("LonnDemo2026!"),
    [persona, setPersona] = useState(0),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  return (
    <div className="welcome">
      <div className="welcome-card">
        <LonnLogo className="login-logo" />
        <p className="eyebrow">ROUTING INFRASTRUCTURE · LOCAL DEMO</p>
        <h1>
          Your intelligence.
          <br />
          Your workspace.
        </h1>
        <label>
          Explore a demo role
          <select
            value={persona}
            onChange={(e) => {
              const i = Number(e.target.value);
              setPersona(i);
              setEmail(personas[i].email);
              setPassword("LonnDemo2026!");
            }}
          >
            {personas.map((p, i) => (
              <option key={p.email} value={i}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <p>{personas[persona].description}</p>
        <form
          onSubmit={async (e) => {
            e.preventDefault();
            setBusy(true);
            setError("");
            try {
              await api("/api/demo/login", {
                method: "POST",
                body: JSON.stringify({ email, password }),
              });
              await onLogin();
            } catch (e) {
              setError(e instanceof Error ? e.message : String(e));
            } finally {
              setBusy(false);
            }
          }}
        >
          <label>
            Email
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="username"
              required
            />
          </label>
          <label>
            Password
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              required
            />
          </label>
          <Button disabled={busy}>
            {busy ? "Signing in…" : "Sign in"}
            <ArrowUpRight size={16} />
          </Button>
        </form>
        <small>
          Demo password: LonnDemo2026! · These are local demonstration accounts.
        </small>
        {error && (
          <div role="alert" className="error">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
