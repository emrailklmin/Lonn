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
import { bestAptitude, radarData } from "../modelAptitudes";
import { shortModel, type Catalog } from "../types";

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
  const data = radarData(models, coverage);
  return (
    <div className={compact ? "radar compact" : "radar"}>
      <div
        className="radar-chart"
        role="img"
        aria-label={`15-task aptitude radar for ${models.map((m) => shortModel(m.id)).join(", ") || "the model pool"}${coverage.length ? ", with eligible pool coverage" : ""}`}
      >
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={data} outerRadius="55%">
            <PolarGrid stroke="#dce4d7" />
            <PolarAngleAxis
              dataKey="axis"
              tick={{ fontSize: 11, fill: "#52664a" }}
            />
            <PolarRadiusAxis
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1]}
              tick={{ fontSize: 9 }}
              axisLine={false}
            />
            <Tooltip
              allowEscapeViewBox={{ x: true, y: true }}
              content={({ active, payload }) => {
                const row = payload?.[0]?.payload as
                  { task: string; label: string } | undefined;
                if (!active || !row) return null;
                return (
                  <div className="radar-tooltip">
                    <strong>{row.label}</strong>
                    {payload?.map((item) => {
                      const isCoverage = item.dataKey === "coverage";
                      const index = Number(
                        String(item.dataKey).replace("model", ""),
                      );
                      const best = isCoverage
                        ? bestAptitude(coverage, row.task)
                        : null;
                      const model = isCoverage ? best?.model : models[index];
                      const aptitude = isCoverage
                        ? best?.aptitude
                        : model?.task_aptitudes[row.task];
                      if (!model || !aptitude) return null;
                      return (
                        <div key={String(item.dataKey)}>
                          {isCoverage
                            ? `Pool: ${shortModel(model.id)}`
                            : shortModel(model.id)}
                          : {aptitude.score.toFixed(3)}
                        </div>
                      );
                    })}
                  </div>
                );
              }}
            />
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
            {models
              .filter((model) => Object.keys(model.task_aptitudes).length > 0)
              .map((model) => {
                const i = models.indexOf(model);
                return (
                  <Radar
                    key={model.id}
                    name={shortModel(model.id)}
                    dataKey={`model${i}`}
                    stroke={colors[i % colors.length]}
                    fill={colors[i % colors.length]}
                    fillOpacity={models.length === 1 ? 0.2 : 0.07}
                    strokeWidth={2}
                  />
                );
              })}
            {!compact && <Legend wrapperStyle={{ fontSize: 11 }} />}
          </RadarChart>
        </ResponsiveContainer>
      </div>
      {models.some((model) => !Object.keys(model.task_aptitudes).length) && (
        <small className="radar-note">
          Aptitude scores unavailable for this model.
        </small>
      )}
    </div>
  );
}
