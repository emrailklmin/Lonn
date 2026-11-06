import assert from "node:assert/strict";
import test from "node:test";
import { benchmarkTopCost, cumulativeSpending } from "../src/spending.ts";
import type { Usage } from "../src/types";
const claude = "openrouter/anthropic/claude-sonnet-5.5";
const flash = "openrouter/deepseek/deepseek-v4.1-flash";
function row(
  day: string,
  provider: number,
  selected: string | null,
  top?: string,
  fee = 0,
): Usage {
  return {
    id: day,
    team_id: null,
    project_id: null,
    application_id: null,
    user_id: null,
    timestamp: `2026-11-${day}T00:00:00Z`,
    source: "live",
    model: selected,
    status: "success",
    input_tokens: 0,
    output_tokens: 0,
    cost_usd: provider + fee,
    latency_ms: 0,
    details: {
      billing: {
        model_cost_usd: provider,
        service_fee_usd: fee,
        total_cost_usd: provider + fee,
      },
      ...(top
        ? {
            baseline: {
              model_id: top,
              ranking: "global_benchmark_v1",
              input_rate_per_m: 0,
              output_rate_per_m: 0,
              model_cost_usd: 999,
              service_fee_usd: 0,
              total_cost_usd: 999,
              input_tokens: 0,
              output_tokens: 0,
            },
          }
        : {}),
    },
  };
}
const approx = (actual: number, expected: number) =>
  assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} != ${expected}`);

test("historical benchmark comparison uses provider cost without scaling recorded fees", () => {
  approx(benchmarkTopCost(row("01", 1.17, flash, claude, 1.28))!, 7.57);
  approx(benchmarkTopCost(row("01", 1.17, flash, flash, 1.28))!, 1.17);
  assert.equal(benchmarkTopCost(row("01", 0, flash, claude)), 0);
});
test("saved estimates remain frozen, including unknown baselines", () => {
  const saved = row("01", 1.17, flash, claude, 1.28);
  saved.details.billing = {
    ...saved.details.billing!,
    service_fee_basis: "estimated_savings_v1",
    estimated_top_model_cost_usd: 7.57,
    estimated_savings_usd: 6.4,
  };
  saved.model = "unknown/future-model";
  assert.equal(benchmarkTopCost(saved), 7.57);
  saved.details.billing.service_fee_basis = "estimated_savings_v2";
  assert.equal(benchmarkTopCost(saved), 7.57);
  saved.details.billing.estimated_top_model_cost_usd = null;
  assert.equal(benchmarkTopCost(saved), null);
});
test("excludes unknowns from all comparison series; separates provider, fee and customer totals", () => {
  const rows = [
    row("02", 1.17, flash, claude, 1.28),
    row("01", 7.57, claude, claude),
    row("01", 99, "unknown/model", claude),
    row("03", 10, flash, "unknown/model"),
    row("04", 10, flash),
    row("05", 10, null, claude),
  ];
  const result = cumulativeSpending(rows);
  assert.equal(result.comparable.length, 2);
  assert.deepEqual(
    result.data.map((d) => d.day),
    ["2026-11-01", "2026-11-02"],
  );
  approx(result.provider, 8.74);
  approx(result.fee, 1.28);
  approx(result.actual, 10.02);
  approx(result.baseline, 15.14);
  approx(result.savings, 6.4);
  approx(result.netSavings, 5.12);
  approx(result.savingsPercent!, (100 * 6.4) / 15.14);
  approx(result.netSavingsPercent!, (100 * 5.12) / 15.14);
  approx(result.data[1].actual, result.data[1].provider + result.data[1].fee);
});
test("historical escalations scale each completed model's provider costs", () => {
  const escalated = row("01", 8.74, claude, claude, 1.748);
  escalated.details.attempt_costs = [
    { model: flash, cost: 1.404, service_fee: 0.234 },
    { model: claude, cost: 9.084, service_fee: 1.514 },
  ];
  approx(benchmarkTopCost(escalated)!, 15.14);
  escalated.details.attempt_costs[0].model = "unknown/model";
  assert.equal(benchmarkTopCost(escalated), null);
});
test("clamps savings per request and subtracts the fee once", () => {
  const negative = cumulativeSpending([row("01", 7.57, claude, flash)]);
  approx(negative.savings, 0);
  approx(negative.fee, 0);
  approx(negative.netSavings, 0);
  const mixed = cumulativeSpending([
    row("01", 7.57, claude, flash),
    row("02", 1.17, flash, claude, 1.28),
  ]);
  approx(mixed.savings, 6.4);
  approx(mixed.netSavings, 5.12);
  approx(mixed.actual, 10.02);
  approx(mixed.data[1].netSavings, 5.12);
  assert.equal(
    cumulativeSpending([row("01", 0, flash, claude)]).savingsPercent,
    null,
  );
  assert.equal(cumulativeSpending([]).savingsPercent, null);
});
test("excludes older task-specific baselines and unknown provider breakdowns", () => {
  const legacy = row("01", 1.17, flash, claude);
  legacy.details.baseline!.ranking = "Existing task router at lambda=0";
  assert.equal(benchmarkTopCost(legacy), null);
  legacy.details.baseline!.ranking = "global_benchmark_v1";
  delete legacy.details.billing;
  assert.equal(benchmarkTopCost(legacy), null);
});
