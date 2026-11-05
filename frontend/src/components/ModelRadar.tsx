import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from "recharts";
import { shortModel, type Catalog } from "../types";
export const attributes = [
  { key: "coding", label: "Coding" },
  { key: "math_reasoning", label: "Math" },
  { key: "creative_writing", label: "Writing" },
  { key: "general_knowledge_extraction", label: "Knowledge" },
];
const colors = ["#356a52", "#b17c45", "#7486b0", "#b16f84"];
export function ModelRadar({
  models,
  coverage = [],
  compact = false,
}: {
  models: Catalog[];
  coverage?: Catalog[];
  compact?: boolean;
}) {
  const data = attributes.map(({ key, label }) => ({
    attribute: label,
    ...Object.fromEntries(
      models.map((m, i) => [`model${i}`, m.topic_aptitudes[key] ?? null]),
    ),
    coverage: coverage.length
      ? Math.max(...coverage.map((m) => m.topic_aptitudes[key] ?? 0))
      : null,
  }));
  return (
    <div
      className={compact ? "radar compact" : "radar"}
      role="img"
      aria-label={`Attribute radar for ${models.map((m) => shortModel(m.id)).join(", ")}${coverage.length ? ", with eligible pool coverage" : ""}`}
    >
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={data} outerRadius={compact ? "65%" : "68%"}>
          <PolarGrid stroke="#dce4d7" />
          <PolarAngleAxis
            dataKey="attribute"
            tick={{ fontSize: 11, fill: "#6f7e68" }}
          />
          <PolarRadiusAxis
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tick={{ fontSize: 9 }}
            axisLine={false}
          />
          <Tooltip />
          {coverage.length > 0 && (
            <Radar
              name="Eligible pool coverage"
              dataKey="coverage"
              stroke="#8a947f"
              fill="#dce5d3"
              fillOpacity={0.12}
              strokeDasharray="5 4"
            />
          )}
          {models.map((model, i) => (
            <Radar
              key={model.id}
              name={shortModel(model.id)}
              dataKey={`model${i}`}
              stroke={colors[i % colors.length]}
              fill={colors[i % colors.length]}
              fillOpacity={models.length === 1 ? 0.2 : 0.07}
              strokeWidth={2}
            />
          ))}
          {!compact && <Legend wrapperStyle={{ fontSize: 11 }} />}
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
