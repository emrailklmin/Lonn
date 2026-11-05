import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { cumulativeSpending } from "./spending";
import { money, type Usage } from "./types";
export function SpendingComparison({ rows }: { rows: Usage[] }) {
  const {
    comparable,
    data,
    actual,
    provider,
    fee,
    baseline,
    savings,
    savingsPercent,
    netSavings,
    netSavingsPercent,
  } = cumulativeSpending(rows);
  const percentage = (value: number | null) =>
    value === null
      ? "Percentage unavailable"
      : `${value.toFixed(1)}% of the estimated baseline`;
  return (
    <article className="panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">CUMULATIVE SPENDING</p>
          <h2>Your routing, compared with the top model</h2>
        </div>
        <span className="badge">ESTIMATED BASELINE</span>
      </div>
      <p>
        The highest-quality eligible model is the baseline. Its estimated cost
        uses the benchmark cost ratio against provider costs, before Lönn fees.
      </p>
      <div className="comparison-metrics">
        <div>
          <small>Provider cost</small>
          <strong>{money(provider)}</strong>
          <small>Recorded tokens × model rates</small>
        </div>
        <div>
          <small>Lönn fees</small>
          <strong>{money(fee)}</strong>
          <small>20% of positive estimated savings per request</small>
        </div>
        <div>
          <small>Total customer spend</small>
          <strong>{money(actual)}</strong>
          <small>Provider costs + Lönn fees</small>
        </div>
        <div>
          <small>Estimated top-model spend</small>
          <strong>{money(baseline)}</strong>
          <small>Benchmark comparison · no Lönn fee</small>
        </div>
        <div>
          <small>Estimated savings before fees</small>
          <strong>{money(savings)}</strong>
          <small>{percentage(savingsPercent)}</small>
        </div>
        <div>
          <small>Estimated savings after fees</small>
          <strong>{money(netSavings)}</strong>
          <small>{percentage(netSavingsPercent)}</small>
        </div>
      </div>
      <div className="chart">
        {data.length ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid vertical={false} stroke="#e3e9de" />
              <XAxis
                dataKey="day"
                tickFormatter={(v) => v.slice(5)}
                fontSize={11}
              />
              <YAxis
                fontSize={10}
                width={62}
                tickFormatter={(v) => "$" + Number(v).toFixed(2)}
              />
              <Tooltip formatter={(value) => money(Number(value))} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line
                type="monotone"
                name="Total customer spend"
                dataKey="actual"
                stroke="#356a52"
                strokeWidth={2.5}
                dot={false}
              />
              <Line
                type="monotone"
                name="Provider cost"
                dataKey="provider"
                stroke="#7b9275"
                strokeDasharray="3 3"
                strokeWidth={1.5}
                dot={false}
              />
              <Line
                type="monotone"
                name="Lönn fees"
                dataKey="fee"
                stroke="#8d82a1"
                strokeWidth={1.5}
                dot={false}
              />
              <Line
                type="monotone"
                name="Estimated top-model spend"
                dataKey="baseline"
                stroke="#b18a5c"
                strokeDasharray="6 4"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="empty">
            No comparable requests yet. New requests preserve their baseline
            automatically.
          </p>
        )}
      </div>
      <small>
        The comparison uses relative costs from the 690-prompt benchmark;
        savings are estimates and may differ for your workload. Each new request
        pays 20% of max(0, estimated top-model cost − provider cost). Missing
        benchmark data means no savings fee. {comparable.length} of{" "}
        {rows.length} requests have comparable baseline and cost data.
        Historical requests retain their recorded fees. Seeded requests are
        synthetic.
      </small>
    </article>
  );
}
