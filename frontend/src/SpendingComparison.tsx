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
import { money } from "./types";

export function SpendingComparison({
  comparison,
  totalRequests,
}: {
  comparison: ReturnType<typeof cumulativeSpending>;
  totalRequests: number;
}) {
  const { comparable, data } = comparison;
  return (
    <article className="panel overview-spending">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">SPENDING</p>
          <h2>Spend comparison</h2>
        </div>
        {comparable.length < totalRequests && (
          <small>
            {comparable.length} of {totalRequests} requests comparable
          </small>
        )}
      </div>
      <div className="chart overview-chart">
        {data.length ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 8, right: 8, left: 0 }}>
              <CartesianGrid vertical={false} stroke="#e8ede5" />
              <XAxis
                dataKey="day"
                tickFormatter={(value) => value.slice(5)}
                fontSize={11}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                fontSize={10}
                width={58}
                tickFormatter={(value) => "$" + Number(value).toFixed(2)}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip formatter={(value) => money(Number(value))} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line
                type="monotone"
                name="Actual customer spend"
                dataKey="actual"
                stroke="#356a52"
                strokeWidth={2.5}
                dot={data.length === 1 ? { r: 3 } : false}
              />
              <Line
                type="monotone"
                name="Estimated top-model spend"
                dataKey="baseline"
                stroke="#b18a5c"
                strokeDasharray="6 4"
                strokeWidth={2}
                dot={data.length === 1 ? { r: 3 } : false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="empty">No comparable requests yet.</p>
        )}
      </div>
    </article>
  );
}
