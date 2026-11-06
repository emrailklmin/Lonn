import "./LonnLogo.css";

export function LonnLogo({ className = "", iconOnly = false, inverse = false }: {
  className?: string;
  iconOnly?: boolean;
  inverse?: boolean;
}) {
  return <span className={`lonn-brand ${inverse ? "lonn-brand-inverse" : ""} ${className}`} aria-label="Lönn AI">
    <span className="lonn-brand-mark" aria-hidden="true"><img src="/brand/lonn-network.png" alt="" /></span>
    {!iconOnly && <span className="lonn-brand-word">Lönn<span>AI</span></span>}
  </span>;
}
