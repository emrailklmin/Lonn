import type { Region } from "../types";
import { worldMapPaths } from "./worldMapPaths";

export const regionLabels: Record<Region, string> = {
  north_america: "North America",
  south_america: "South America",
  europe: "Europe",
  asia: "Asia",
  africa: "Africa",
  oceania: "Oceania",
};

export function RegionMap({
  allowed,
  onChange,
  disabled = false,
}: {
  allowed: Region[];
  onChange: (regions: Region[]) => void;
  disabled?: boolean;
}) {
  function toggle(region: Region) {
    if (disabled) return;
    onChange(
      allowed.includes(region)
        ? allowed.filter((item) => item !== region)
        : [...allowed, region],
    );
  }
  return (
    <div className="region-map">
      <div className="region-map-heading">
        <strong>Allowed regions</strong>
        <span>{allowed.length} of 6 allowed</span>
      </div>
      <p>Click a continent to allow or restrict it.</p>
      <div className="region-legend" aria-label="Map legend">
        <span>
          <i className="allowed" /> Allowed
        </span>
        <span>
          <i /> Restricted
        </span>
      </div>
      <svg viewBox="0 0 960 387" aria-label="Allowed regions world map">
        {(Object.keys(regionLabels) as Region[]).map((region) => {
          const selected = allowed.includes(region);
          return (
            <g
              key={region}
              className={selected ? "continent allowed" : "continent"}
              role="button"
              tabIndex={disabled ? -1 : 0}
              aria-label={regionLabels[region]}
              aria-pressed={selected}
              aria-disabled={disabled}
              onClick={() => toggle(region)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  toggle(region);
                }
              }}
            >
              <title>{`${regionLabels[region]}: ${selected ? "allowed" : "restricted"}`}</title>
              <path d={worldMapPaths[region]} fillRule="evenodd" />
            </g>
          );
        })}
      </svg>
      <div className="region-choices" aria-label="Region selection">
        {(Object.keys(regionLabels) as Region[]).map((region) => (
          <button
            key={region}
            type="button"
            className={
              allowed.includes(region)
                ? "region-choice allowed"
                : "region-choice"
            }
            aria-pressed={allowed.includes(region)}
            disabled={disabled}
            onClick={() => toggle(region)}
          >
            <span aria-hidden="true" />
            {regionLabels[region]}
          </button>
        ))}
      </div>
      {!allowed.length && (
        <p className="region-empty" role="status">
          No regions allowed. No models will be eligible.
        </p>
      )}
    </div>
  );
}
