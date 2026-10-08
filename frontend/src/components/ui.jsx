import React, { useEffect, useState } from "react";
import { AlertTriangle, Inbox } from "lucide-react";
import { errorMessage } from "../api/client";

export const CATEGORY_LABELS = {
  correct_solution: "Correct solution",
  arithmetic_error: "Arithmetic slip",
  sign_error: "Sign error",
  incorrect_operation: "Wrong operation",
  algebraic_manipulation_error: "Moving terms incorrectly",
  skipped_step: "Skipped step",
  final_answer_only_error: "Final answer mismatch",
  transcription_uncertainty: "Couldn't read clearly",
  insufficient_information: "Not enough to check",
  formula_selection_error: "Wrong formula",
};

export const pct = (v, digits = 0) => `${((v ?? 0) * 100).toFixed(digits)}%`;
export const fmt = (v, digits = 3) => (v === null || v === undefined ? "—" : Number(v).toFixed(digits));

export function toneFor(value) {
  if (value >= 0.8) return "good";
  if (value >= 0.5) return "warn";
  return "bad";
}

export function Card({ title, subtitle, actions, children, className = "", ...rest }) {
  return (
    <section className={`card ${className}`} {...rest}>
      {(title || actions) && (
        <div className="card-head">
          <div>
            {title && <h2>{title}</h2>}
            {subtitle && <div className="sub">{subtitle}</div>}
          </div>
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

export function PageHead({ title, description, children }) {
  return (
    <div className="page-head">
      <div>
        <h1>{title}</h1>
        {description && <p>{description}</p>}
      </div>
      {children && <div className="row wrap">{children}</div>}
    </div>
  );
}

export function Badge({ tone, children, icon: Icon }) {
  return (
    <span className={`badge ${tone || ""}`}>
      {Icon && <Icon size={13} />}
      {children}
    </span>
  );
}

export function CategoryBadge({ category }) {
  if (!category) return null;
  const tone =
    category === "correct_solution" ? "good" : ["insufficient_information", "transcription_uncertainty"].includes(category) ? "warn" : "bad";
  return <Badge tone={tone}>{CATEGORY_LABELS[category] || category.replaceAll("_", " ")}</Badge>;
}

export function Meter({ value, tone, label }) {
  return (
    <div className={`meter ${tone || ""}`} role="meter" aria-valuenow={Math.round((value || 0) * 100)} aria-valuemin={0} aria-valuemax={100} aria-label={label}>
      <span style={{ width: `${Math.max(0, Math.min(1, value || 0)) * 100}%` }} />
    </div>
  );
}

export function Stat({ label, value, hint }) {
  return (
    <div className="card tight stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
      {hint && <span className="faint">{hint}</span>}
    </div>
  );
}

export function Skeleton({ height = 18, width = "100%", style }) {
  return <div className="skeleton" style={{ height, width, ...style }} />;
}

export function LoadingCard({ lines = 4 }) {
  return (
    <div className="card stack">
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton key={i} width={`${90 - i * 12}%`} />
      ))}
    </div>
  );
}

export function ErrorState({ error, onRetry }) {
  return (
    <div className="alert bad" role="alert">
      <AlertTriangle size={18} />
      <div className="stack" style={{ gap: 6 }}>
        <span>{typeof error === "string" ? error : errorMessage(error)}</span>
        {onRetry && (
          <button className="btn sm secondary" onClick={onRetry} style={{ alignSelf: "flex-start" }}>
            Try again
          </button>
        )}
      </div>
    </div>
  );
}

export function Empty({ icon: Icon = Inbox, title, children }) {
  return (
    <div className="empty">
      <Icon size={30} />
      <strong>{title}</strong>
      {children}
    </div>
  );
}

export function Tabs({ tabs, value, onChange }) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map(({ id, label, icon: Icon }) => (
        <button key={id} role="tab" aria-selected={value === id} className={`tab ${value === id ? "active" : ""}`} onClick={() => onChange(id)}>
          {Icon && <Icon size={15} />}
          {label}
        </button>
      ))}
    </div>
  );
}

/** Minimal data-fetching hook: { data, error, loading, reload, setData }.
 * `loading` is true until the request for the *current* deps has settled, so
 * callers never render stale-null data in the frame after deps change. */
export function useFetch(fn, deps) {
  const [tick, setTick] = useState(0);
  const key = JSON.stringify([...deps, tick]);
  const [state, setState] = useState({ data: null, error: null, key: null });
  useEffect(() => {
    let alive = true;
    Promise.resolve(fn())
      .then((res) => alive && setState({ data: res?.data ?? null, error: null, key }))
      .catch((error) => alive && setState((s) => ({ data: s.data, error, key })));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return {
    data: state.data,
    error: state.key === key ? state.error : null,
    loading: state.key !== key,
    reload: () => setTick((t) => t + 1),
    setData: (data) => setState((s) => ({ ...s, data })),
  };
}

/** Heat colour from 0..1 using the current theme's good/warn/bad tokens via color-mix. */
export function heat(value) {
  const v = Math.max(0, Math.min(1, value ?? 0));
  const tone = v >= 0.8 ? "--good" : v >= 0.5 ? "--warn" : "--bad";
  const strength = Math.round(18 + 50 * (v >= 0.8 ? v : v >= 0.5 ? 0.7 : 1 - v));
  return { background: `color-mix(in srgb, var(${tone}) ${strength}%, var(--surface))`, color: "var(--text)" };
}

/** Keeps one broken page from blanking the whole app; resets on navigation via `key`. */
export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return (
        <div className="alert bad" role="alert">
          <AlertTriangle size={18} />
          <div className="stack" style={{ gap: 4 }}>
            <strong>This page hit an error.</strong>
            <span className="mono small">{String(this.state.error?.message || this.state.error)}</span>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
