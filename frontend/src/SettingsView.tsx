import {
  DEFAULT_LAMBDA,
  MAX_LAMBDA,
  LambdaSlider,
} from "./components/LambdaSlider";
import { ModelRadar } from "./components/ModelRadar";
import { RegionMap } from "./components/RegionMap";
import { useState } from "react";
import { Button } from "./components/ui/button";
import { CopyButton } from "./components/CopyButton";
import {
  api,
  shortModel,
  type Account,
  type Catalog,
  type Settings,
} from "./types";

const requirements = [
  { key: "require_zdr", label: "Zero data retention" },
  { key: "forbid_training", label: "No provider training" },
  { key: "require_encryption", label: "Encryption" },
  { key: "require_private_network", label: "Private network" },
  { key: "require_audit_logs", label: "Audit logs" },
] as const;

export function SettingsView({
  account,
  catalog,
  refresh,
}: {
  account: Account;
  catalog: Catalog[];
  refresh: () => Promise<void>;
}) {
  const [settings, setSettings] = useState<Settings>(account.settings),
    [key, setKey] = useState(""),
    [message, setMessage] = useState(""),
    [busy, setBusy] = useState(false),
    [showKey, setShowKey] = useState(false);
  async function action(fn: () => Promise<unknown>, text: string) {
    setBusy(true);
    setMessage("");
    try {
      await fn();
      await refresh();
      setMessage(text);
    } catch (e) {
      setMessage(String(e));
    } finally {
      setBusy(false);
    }
  }
  async function savePolicy(patch: Partial<Settings>) {
    const previous = settings;
    setSettings((current) => ({ ...current, ...patch }));
    setBusy(true);
    setMessage("");
    let saved: Settings | undefined;
    try {
      saved = await api<Settings>("/api/settings");
      await api("/api/settings", {
        method: "PUT",
        body: JSON.stringify({ ...saved, ...patch }),
      });
      await refresh();
      setMessage("Security settings saved.");
    } catch (e) {
      const original = saved ?? previous;
      setSettings((current) => {
        const restored = { ...current };
        for (const key of Object.keys(patch) as (keyof Settings)[]) {
          Object.assign(restored, { [key]: original[key] });
        }
        return restored;
      });
      setMessage(`Could not save settings: ${String(e)}`);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="settings-grid">
      <article className="panel">
        <h2>Routing preferences</h2>
        <p>
          Choose a quality preference for the router. Higher λ gives quality
          more weight; lower λ favors lower cost.
        </p>
        <div className="quality-control">
          <label htmlFor="routing-lambda">
            Routing preference λ <output>{settings.lam.toFixed(2)}</output>
          </label>
          <LambdaSlider
            id="routing-lambda"
            value={settings.lam}
            onChange={(lam) => setSettings({ ...settings, lam })}
          />
          <div className="slider-scale">
            <span>0 · Cost only</span>
            <span>{MAX_LAMBDA} · Quality only</span>
          </div>
          <button
            className="balance-preset"
            onClick={() => setSettings({ ...settings, lam: DEFAULT_LAMBDA })}
          >
            {DEFAULT_LAMBDA} · Balanced default
          </button>
          <small>
            The curve illustrates the suggested balance. It does not represent
            measured quality or change the λ value sent to the router.
          </small>
        </div>
        {account.user.role === "enterprise_admin" && (
          <label className="check">
            <input
              type="checkbox"
              checked={!!settings.enforce_lam}
              onChange={(e) =>
                setSettings({ ...settings, enforce_lam: e.target.checked })
              }
            />{" "}
            Enforce this λ across the organization
          </label>
        )}
        <small>
          When enforcement is off, team, project, application, and personal
          preferences can take precedence.
        </small>
        <h2>Security & privacy</h2>
        <p>
          Model security and regions are demo assumptions. Provider safeguards
          and processing locations are not verified or controlled by this demo.
        </p>
        <p role="status">
          {catalog.filter((model) => model.eligible).length} of {catalog.length}{" "}
          models eligible
        </p>
        <RegionMap
          allowed={settings.allowed_regions}
          disabled={busy}
          onChange={(allowed_regions) => void savePolicy({ allowed_regions })}
        />
        <h3>Required safeguards</h3>
        {requirements.map(({ key, label }) => (
          <label className="check" key={key}>
            <input
              type="checkbox"
              disabled={busy}
              checked={settings[key]}
              onChange={(event) =>
                void savePolicy({ [key]: event.target.checked })
              }
            />{" "}
            {label}
          </label>
        ))}
        <h2>Model selection</h2>
        {catalog
          .filter((m) => m.active)
          .map((m) => (
            <label
              key={m.id}
              className={
                m.policy_eligible
                  ? "check model-choice"
                  : "check model-choice unavailable"
              }
            >
              <input
                type="checkbox"
                disabled={busy || !m.policy_eligible}
                checked={
                  m.policy_eligible && !settings.disabled_models.includes(m.id)
                }
                onChange={(e) =>
                  void savePolicy({
                    disabled_models: e.target.checked
                      ? settings.disabled_models.filter((id) => id !== m.id)
                      : [...settings.disabled_models, m.id],
                  })
                }
              />
              <span>
                {shortModel(m.id)}
                {!m.policy_eligible && (
                  <small>{m.policy_exclusion_reasons.join(" · ")}</small>
                )}
              </span>
            </label>
          ))}
        <details className="draft-coverage">
          <summary>Preview selected model coverage</summary>
          <ModelRadar
            models={[]}
            coverage={catalog.filter(
              (m) => m.eligible && !settings.disabled_models.includes(m.id),
            )}
          />
          <small>
            Best router aptitude per task among models eligible for your
            settings.
          </small>
        </details>
        <Button
          disabled={busy}
          onClick={() =>
            void action(async () => {
              await api("/api/settings", {
                method: "PUT",
                body: JSON.stringify(settings),
              });
              if (!settings.enforce_lam)
                await api("/api/preference", {
                  method: "PUT",
                  body: JSON.stringify({ lam: settings.lam }),
                });
            }, "Routing and privacy settings saved.")
          }
        >
          Save preferences
        </Button>
      </article>
      <div>
        <article className="panel">
          <h2>Your Lönn API</h2>
          <p>
            Your personal key identifies you. Specify an application you can
            access.
          </p>
          <label>
            API endpoint<code>{account.endpoint}</code>
          </label>
          <label>
            Personal API key
            <code>
              {showKey ? account.api_key : "lonn_personal_••••••••••••••••"}
            </code>
          </label>
          <div className="actions">
            <Button variant="outline" onClick={() => setShowKey(!showKey)}>
              {showKey ? "Hide" : "Reveal"}
            </Button>
            <CopyButton value={account.api_key ?? ""} />
            <Button
              variant="ghost"
              disabled={busy}
              onClick={() =>
                void action(
                  () => api("/api/key/rotate", { method: "POST" }),
                  "Key rotated. Update your customer application.",
                )
              }
            >
              Rotate key
            </Button>
          </div>
          <small>Rotating immediately invalidates the previous key.</small>
          <h3>Python integration</h3>
          <pre>{`from lonnrouter import LonnRouter\n\nwith LonnRouter(\n    api_key="YOUR_LONN_API_KEY",\n    application="engineering-app",\n    base_url="${account.endpoint}"\n) as client:\n    result = client.generate(\n        "Explain how neural networks learn."\n    )\n    print(result.content)`}</pre>
          <CopyButton
            value={`from lonnrouter import LonnRouter\n\nwith LonnRouter(api_key="YOUR_LONN_API_KEY", application="engineering-app", base_url="${account.endpoint}") as client:\n    print(client.generate("Explain how neural networks learn.").content)`}
          />
        </article>
        {account.user.role === "enterprise_admin" && (
          <article className="panel">
            <h2>Presenter configuration</h2>
            <span className={account.provider_configured ? "status" : "badge"}>
              {account.provider_configured
                ? "Credential stored"
                : "Credential needed"}
            </span>
            <p>
              Your private OpenRouter credential is stored only on this
              computer, separately from the account database. It is never
              returned by the API.
            </p>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                void action(async () => {
                  await api("/api/provider", {
                    method: "PUT",
                    body: JSON.stringify({ key }),
                  });
                  setKey("");
                }, "Provider credential saved.");
              }}
            >
              <label>
                OpenRouter API key
                <input
                  type="password"
                  autoComplete="off"
                  value={key}
                  onChange={(e) => setKey(e.target.value)}
                  placeholder="sk-or-…"
                  required
                />
              </label>
              <div className="actions">
                <Button disabled={busy || !key}>Save credential</Button>
                <Button
                  type="button"
                  variant="ghost"
                  disabled={busy}
                  onClick={() =>
                    void action(
                      () => api("/api/provider", { method: "DELETE" }),
                      "Provider credential removed.",
                    )
                  }
                >
                  Remove
                </Button>
              </div>
            </form>
          </article>
        )}
      </div>
      {message && (
        <div className="notice feedback" role="status">
          {message}
        </div>
      )}
    </div>
  );
}
