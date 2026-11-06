import assert from "node:assert/strict";
import test from "node:test";
import { attributes, bestAptitude, radarData } from "../src/modelAptitudes.ts";
import type { Catalog, TaskAptitude } from "../src/types";

const measured: TaskAptitude = {
  score: 0.7,
  source: "measured",
  sample_size: 42,
  cost_usd: 0.1,
  cost_source: "task_benchmark",
};
const supplied: TaskAptitude = {
  score: 0.9,
  source: "supplied",
  sample_size: null,
  cost_usd: 0.2,
  cost_source: "global_benchmark",
};
const model = (id: string, task_aptitudes: Record<string, TaskAptitude>) =>
  ({ id, task_aptitudes }) as Catalog;

test("radars expose each router task exactly once and map the API scores", () => {
  assert.deepEqual(
    new Set(attributes.map((item) => item.key)),
    new Set([
      "math_problem",
      "code_function",
      "code_algorithmic",
      "science_expert",
      "knowledge_quantitative",
      "knowledge_recall",
      "instruction_following",
      "business_writing",
      "summarization",
      "translation",
      "information_extraction",
      "creative_writing",
      "conversation",
      "tool_calling",
      "other",
    ]),
  );
  const first = model("first", {
    math_problem: measured,
    creative_writing: supplied,
  });
  const second = model("second", { math_problem: { ...measured, score: 0 } });
  const rows = radarData([first, second], [first, second]);
  assert.equal(rows.length, 15);
  assert.deepEqual(
    rows.find((row) => row.task === "math_problem"),
    {
      task: "math_problem",
      label: "Math problems",
      axis: "Math",
      model0: 0.7,
      model1: 0,
      coverage: 0.7,
    },
  );
  const creative = rows.find((row) => row.task === "creative_writing")!;
  assert.equal(creative.model0, 0.9);
  assert.equal(creative.model1, null);
  assert.equal(creative.coverage, 0.9);
  assert.equal(bestAptitude([second], "creative_writing"), null);
});
