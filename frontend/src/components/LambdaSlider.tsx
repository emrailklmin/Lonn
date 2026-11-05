export function LambdaSlider({
  id,
  value,
  onChange,
  disabled = false,
  label = "Routing preference λ",
}: {
  id: string;
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  label?: string;
}) {
  return (
    <div className="curve-slider">
      <svg viewBox="0 0 400 170" preserveAspectRatio="none" aria-hidden="true">
        <line x1="20" y1="145" x2="380" y2="145" className="curve-axis" />
        <line x1="20" y1="145" x2="20" y2="20" className="curve-axis" />
        <polyline
          className="curve-line"
          points={Array.from({ length: 101 }, (_, i) => {
            const x = i / 100;
            return `${20 + 360 * x},${145 - (115 * Math.log1p(30 * x)) / Math.log1p(30)}`;
          }).join(" ")}
        />
        <circle
          className="curve-elbow"
          cx="92"
          cy={145 - (115 * Math.log1p(6)) / Math.log1p(30)}
          r="4"
        />
        <circle
          className="curve-handle"
          cx={20 + 360 * value}
          cy={145 - (115 * Math.log1p(30 * value)) / Math.log1p(30)}
          r="8"
        />
      </svg>
      <input
        id={id}
        aria-label={label}
        aria-valuetext={`λ ${value}`}
        type="range"
        min="0"
        max="1"
        step="any"
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}
