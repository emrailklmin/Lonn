import type { Usage } from "./types";
// Legacy records predate saved benchmark estimates. Preserve the rounded
// benchmark snapshot used for their original comparison and fees.
const benchmarkCosts: Record<string, number> = {
  "openrouter/anthropic/claude-sonnet-5.5": 7.57,
  "openrouter/meta/muse-spark-1.3": 7.37,
  "openrouter/google/gemini-3.8-flash": 4.53,
  "openrouter/deepseek/deepseek-v4-pro": 3.98,
  "openrouter/deepseek/deepseek-v4.1-flash": 1.17,
  "openrouter/z-ai/glm-5.3-flash": 0.38,
  "openrouter/openai/gpt-oss-120b": 0.3,
  "openrouter/openai/gpt-oss-20b": 0.19,
};
const finite = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);
export const providerCost = (row: Usage): number | null =>
  finite(row.details.billing?.model_cost_usd)
    ? row.details.billing!.model_cost_usd
    : null;
export const lonnFee = (row: Usage): number | null =>
  finite(row.details.billing?.service_fee_usd)
    ? row.details.billing!.service_fee_usd
    : null;

export function benchmarkTopCost(row: Usage): number | null {
  const billing = row.details.billing;
  if (
    billing?.service_fee_basis === "estimated_savings_v1" ||
    billing?.service_fee_basis === "estimated_savings_v2"
  ) {
    // New requests freeze their estimate. Never reconstruct an unknown baseline.
    return finite(billing.estimated_top_model_cost_usd)
      ? billing.estimated_top_model_cost_usd
      : null;
  }
  if (row.details.baseline?.ranking !== "global_benchmark_v1") return null;
  const top =
    row.details.baseline.benchmark_cost_per_690 ??
    benchmarkCosts[row.details.baseline.model_id];
  if (!finite(top)) return null;
  const attempts = row.details.attempt_costs;
  if (attempts?.length) {
    let estimate = 0;
    for (const attempt of attempts) {
      const selected =
        attempt.benchmark_cost_per_690 ?? benchmarkCosts[attempt.model];
      const modelCost =
        attempt.model_cost_usd ?? attempt.cost - attempt.service_fee;
      if (!finite(selected) || selected <= 0 || !finite(modelCost)) return null;
      estimate += (modelCost * top) / selected;
    }
    return estimate;
  }
  const selected = row.model && benchmarkCosts[row.model],
    cost = providerCost(row);
  return finite(selected) && selected > 0 && cost !== null
    ? (cost * top) / selected
    : null;
}

export function cumulativeSpending(rows: Usage[]) {
  const comparable = rows.filter(
    (row) =>
      benchmarkTopCost(row) !== null &&
      providerCost(row) !== null &&
      lonnFee(row) !== null,
  );
  const daily: Record<
    string,
    {
      day: string;
      actual: number;
      provider: number;
      fee: number;
      baseline: number;
      savings: number;
      netSavings: number;
    }
  > = {};
  for (const row of comparable) {
    const day = row.timestamp.slice(0, 10),
      d = (daily[day] ??= {
        day,
        actual: 0,
        provider: 0,
        fee: 0,
        baseline: 0,
        savings: 0,
        netSavings: 0,
      });
    d.actual += row.cost_usd;
    d.provider += providerCost(row)!;
    d.fee += lonnFee(row)!;
    d.baseline += benchmarkTopCost(row)!;
    const savings = Math.max(0, benchmarkTopCost(row)! - providerCost(row)!);
    d.savings += savings;
    d.netSavings += savings - lonnFee(row)!;
  }
  let actual = 0,
    provider = 0,
    fee = 0,
    baseline = 0,
    savings = 0,
    netSavings = 0;
  const data = Object.values(daily)
    .sort((a, b) => a.day.localeCompare(b.day))
    .map((d) => ({
      day: d.day,
      actual: (actual += d.actual),
      provider: (provider += d.provider),
      fee: (fee += d.fee),
      baseline: (baseline += d.baseline),
      savings: (savings += d.savings),
      netSavings: (netSavings += d.netSavings),
    }));
  return {
    comparable,
    data,
    actual,
    provider,
    fee,
    baseline,
    savings,
    savingsPercent: baseline > 0 ? (100 * savings) / baseline : null,
    netSavings,
    netSavingsPercent: baseline > 0 ? (100 * netSavings) / baseline : null,
  };
}
