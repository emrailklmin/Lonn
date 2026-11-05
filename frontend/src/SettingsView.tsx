import { LambdaSlider } from "./components/LambdaSlider";
import { ModelRadar } from "./components/ModelRadar";
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
  return (
    <div className="settings-grid">
      <article className="panel">
        <h2>Routing preferences</h2>
        <p>
          Choose a cost weight for the router. Higher λ gives cost more weight;
          lower λ favors expected accuracy.
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
            <span>0 · More accuracy weight</span>
            <span>1 · More cost weight</span>
          </div>
          <button
            className="balance-preset"
            onClick={() => setSettings({ ...settings, lam: 0.2 })}
          >
            0.20 · Suggested balance
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
          Unknown metadata fails strict requirements. The current catalog has no
          verified endpoint guarantees; these restrictions will exclude all
          models.
        </p>
        <label className="check">
          <input
            type="checkbox"
            checked={settings.require_zdr}
            onChange={(e) =>
              setSettings({ ...settings, require_zdr: e.target.checked })
            }
          />{" "}
          Require zero data retention
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={settings.forbid_training}
            onChange={(e) =>
              setSettings({ ...settings, forbid_training: e.target.checked })
            }
          />{" "}
          Prohibit provider training on data
        </label>
        <label>
          Processing region
          <select
            value={settings.region}
            onChange={(e) =>
              setSettings({ ...settings, region: e.target.value })
            }
          >
            <option value="any">No restriction</option>
            <option value="EU">EU only</option>
            <option value="US">US only</option>
          </select>
        </label>
        <label>
          Maximum retention days (blank = no restriction)
          <input
            type="number"
            min="0"
            max="3650"
            value={settings.max_retention_days ?? ""}
            onChange={(e) =>
              setSettings({
                ...settings,
                max_retention_days:
                  e.target.value === "" ? null : Number(e.target.value),
              })
            }
          />
        </label>
        <h2>Model selection</h2>
        {catalog
          .filter((m) => m.active)
          .map((m) => (
            <label key={m.id} className="check">
              <input
                type="checkbox"
                checked={!settings.disabled_models.includes(m.id)}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    disabled_models: e.target.checked
                      ? settings.disabled_models.filter((id) => id !== m.id)
                      : [...settings.disabled_models, m.id],
                  })
                }
              />
              {shortModel(m.id)}
            </label>
          ))}
        <details className="draft-coverage">
          <summary>Preview selected model coverage</summary>
          <ModelRadar
            models={[]}
            coverage={catalog.filter(
              (m) => m.active && !settings.disabled_models.includes(m.id),
            )}
          />
          <small>
            Best catalog score per category among checked models, before privacy
            filtering. Explore individual profiles and comparisons on the Model
            pool page.
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
