export type Settings = {
  lam: number;
  enforce_lam: boolean;
  require_zdr: boolean;
  forbid_training: boolean;
  region: string;
  max_retention_days: number | null;
  disabled_models: string[];
};
export type Account = {
  id: string;
  name: string;
  api_key: string | null;
  user: {
    id: string;
    name: string;
    email: string;
    role: "enterprise_admin" | "enterprise_user" | "private_user";
    team_id: string | null;
    team_name: string | null;
  };
  settings: Settings;
  provider_configured: boolean;
  endpoint: string;
};
export type Catalog = {
  id: string;
  provider: string;
  active: boolean;
  eligible: boolean;
  availability: string;
  exclusion_reasons: string[];
  privacy_status: string;
  cost_per_m_input: number;
  cost_per_m_output: number;
  context_window: number;
  price_basis: string;
  aptitude_basis: string;
  topic_aptitudes: Record<string, number>;
};
export type Billing = {
  model_cost_usd: number;
  service_fee_usd: number;
  total_cost_usd: number;
  service_fee_rate?: number;
  service_fee_basis?: string;
  estimated_top_model_cost_usd?: number | null;
  estimated_savings_usd?: number | null;
};
export type Usage = {
  id: string;
  team_id: string | null;
  project_id: string | null;
  application_id: string | null;
  user_id: string | null;
  timestamp: string;
  source: string;
  model: string | null;
  status: string;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  latency_ms: number;
  details: Record<string, unknown> & {
    baseline?: {
      model_id: string;
      ranking?: string;
      benchmark_cost_per_690?: number | null;
      input_rate_per_m: number;
      output_rate_per_m: number;
      estimated_cost_usd?: number | null;
      model_cost_usd?: number;
      service_fee_usd?: number;
      total_cost_usd?: number;
      input_tokens: number;
      output_tokens: number;
    };
    billing?: Billing;
    attempt_costs?: {
      model: string;
      cost: number;
      model_cost_usd?: number;
      service_fee: number;
      benchmark_cost_per_690?: number | null;
    }[];
  };
};
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}
export async function api<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new ApiError(
      typeof data.detail === "string"
        ? data.detail
        : `Request failed (${response.status})`,
      response.status,
    );
  }
  return response.json();
}
export const shortModel = (id: string | null) =>
  id?.split("/").slice(2).join("/") || "No selection";
export const money = (n: number) =>
  "$" + n.toFixed(n !== 0 && Math.abs(n) < 0.0001 ? 8 : 4);
export type Team = {
  id: string;
  customer_id: string;
  name: string;
  enforced_lam: number | null;
  enabled: boolean;
  disabled_models: string[];
  eligible_model_ids: string[];
};
export type DemoUser = {
  id: string;
  customer_id: string;
  name: string;
  email: string;
  role: "enterprise_admin" | "enterprise_user" | "private_user";
  team_id: string | null;
  enabled: boolean;
};
