import { CopyButton } from "./components/CopyButton";
import { api, type Account } from "./types";
import { Button } from "./components/ui/button";
import { useState } from "react";
export function AccessView({
  account,
  refresh,
}: {
  account: Account;
  refresh: () => Promise<void>;
}) {
  const [message, setMessage] = useState("");
  return (
    <>
      <div className="notice">
        <div>
          <strong>Managed by your enterprise administrator</strong>
          <p>
            You belong to {account.user.team_name}. Organization and team model
            restrictions apply. Your effective routing preference depends on the
            selected application.
          </p>
        </div>
      </div>
      <article className="panel">
        <h2>Your team integration</h2>
        <p>
          Use your personal key and select an authorized application. Requests
          are attributed to {account.user.team_name}.
        </p>
        <code>{account.endpoint}</code>
        {account.api_key ? (
          <>
            <code>lonn_personal_••••••••••••••••</code>
            <div className="actions">
              <CopyButton value={account.api_key} />
              <Button
                variant="outline"
                onClick={async () => {
                  try {
                    await api("/api/key/rotate", { method: "POST" });
                    await refresh();
                    setMessage("Personal key rotated.");
                  } catch (e) {
                    setMessage(String(e));
                  }
                }}
              >
                Rotate personal key
              </Button>
            </div>
            <small role="status">{message}</small>
          </>
        ) : (
          <p>Team API access is currently revoked.</p>
        )}
        <pre>{`from lonnrouter import LonnRouter\n\nwith LonnRouter(api_key="YOUR_PERSONAL_KEY", application="engineering-app") as client:\n    print(client.generate("Explain binary search").content)`}</pre>
      </article>
      <article className="panel">
        <h2>Privacy requirements</h2>
        <p>
          Zero data retention:{" "}
          {account.settings.require_zdr ? "Required" : "No restriction"}
          <br />
          Provider training:{" "}
          {account.settings.forbid_training ? "Prohibited" : "No restriction"}
          <br />
          Processing region: {account.settings.region}
        </p>
        <small>
          Inspect effective λ and set your personal preference under Projects &
          apps.
        </small>
      </article>
    </>
  );
}
