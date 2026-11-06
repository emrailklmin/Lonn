import type { Catalog, TaskAptitude } from "./types";

// Presentation labels for the router's task IDs. Scores come only from the API.
export const attributes = [
  { key: "math_problem", label: "Math problems", axis: "Math" },
  { key: "code_function", label: "Function coding", axis: "Functions" },
  { key: "code_algorithmic", label: "Algorithmic coding", axis: "Algorithms" },
  { key: "science_expert", label: "Expert science", axis: "Science" },
  {
    key: "knowledge_quantitative",
    label: "Quantitative knowledge",
    axis: "Quantitative",
  },
  {
    key: "knowledge_recall",
    label: "Factual knowledge and recall",
    axis: "Recall",
  },
  {
    key: "instruction_following",
    label: "Instruction following",
    axis: "Instructions",
  },
  { key: "business_writing", label: "Business writing", axis: "Business" },
  { key: "summarization", label: "Summarization", axis: "Summarize" },
  { key: "translation", label: "Translation", axis: "Translation" },
  {
    key: "information_extraction",
    label: "Information extraction",
    axis: "Extraction",
  },
  { key: "creative_writing", label: "Creative writing", axis: "Creative" },
  { key: "conversation", label: "Conversation", axis: "Conversation" },
  { key: "tool_calling", label: "Tool calling", axis: "Tool calling" },
  { key: "other", label: "Other", axis: "Other" },
] as const;

export function bestAptitude(models: Catalog[], task: string) {
  return models.reduce<{ model: Catalog; aptitude: TaskAptitude } | null>(
    (best, model) => {
      const aptitude = model.task_aptitudes[task];
      return aptitude && (!best || aptitude.score > best.aptitude.score)
        ? { model, aptitude }
        : best;
    },
    null,
  );
}

export function radarData(models: Catalog[], coverage: Catalog[]) {
  return attributes.map(({ key, label, axis }) => ({
    task: key,
    label,
    axis,
    ...Object.fromEntries(
      models.map((model, index) => [
        `model${index}`,
        model.task_aptitudes[key]?.score ?? null,
      ]),
    ),
    coverage: bestAptitude(coverage, key)?.aptitude.score ?? null,
  }));
}
