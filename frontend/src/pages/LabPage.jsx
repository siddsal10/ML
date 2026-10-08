import React from "react";
import { Link } from "react-router-dom";
import { ArrowRight, BrainCircuit, GitBranch } from "lucide-react";
import { endpoints } from "../api/client";
import { Badge, ErrorState, LoadingCard, PageHead, fmt, useFetch } from "../components/ui.jsx";

const PIPELINE = [
  ["Photo / drawing", "input"],
  ["OpenCV clean-up & segmentation", "classical CV"],
  ["Symbol recognizer", "SVM · HOG features"],
  ["SymPy step verification", "exact maths"],
  ["Mistake classifier", "Random Forest · 33 features"],
  ["Knowledge tracing", "BKT · PFA · IRT"],
  ["Adaptive practice + teacher groups", "recommender · K-Means"],
];

export default function LabPage() {
  const { data, error, loading, reload } = useFetch(() => endpoints.labModels(), []);
  return (
    <>
      <PageHead title="ML Lab" description="Every learned component of StepWise, how it was trained, and how it compares with alternatives and baselines. All numbers come from the generated training reports." />

      <div className="card">
        <div className="card-head"><h2>Pipeline</h2><Badge tone="accent" icon={GitBranch}>classical ML only</Badge></div>
        <div className="stepper">
          {PIPELINE.map(([name, kind], i) => (
            <React.Fragment key={name}>
              <span className={`step-chip ${kind.includes("·") ? "active" : ""}`} title={kind}>{name} <span className="faint">· {kind}</span></span>
              {i < PIPELINE.length - 1 && <ArrowRight size={14} className="muted" style={{ alignSelf: "center" }} />}
            </React.Fragment>
          ))}
        </div>
      </div>

      {loading && !data ? <LoadingCard /> : error ? <ErrorState error={error} onRetry={reload} /> : (
        <div className="grid grid-2">
          {data.map((m) => (
            <Link key={m.name} to={`/lab/${m.name}`} className="card model-card">
              <div className="row between">
                <span className="row"><BrainCircuit size={18} color="var(--accent)" /><h2>{m.title}</h2></span>
                {m.active_version && <Badge tone="accent">{m.active_version}</Badge>}
              </div>
              <span className="faint">{m.stage} · {m.kind}</span>
              {m.headline ? (
                <div className="row" style={{ alignItems: "baseline", gap: 12 }}>
                  <span className="stat-value">{fmt(m.headline.value)}</span>
                  <span className="muted small">{m.headline.label}{m.headline.baseline !== undefined && <> · {m.headline.baseline_label || "baseline"} {fmt(m.headline.baseline)}</>}</span>
                </div>
              ) : <Badge tone="warn">Not trained — run python -m app.ml.train</Badge>}
              <div className="chips">{m.algorithms.map((a) => <Badge key={a}>{a}</Badge>)}</div>
              <span className="row small" style={{ color: "var(--accent)", fontWeight: 600 }}>Details <ArrowRight size={14} /></span>
            </Link>
          ))}
        </div>
      )}
    </>
  );
}
