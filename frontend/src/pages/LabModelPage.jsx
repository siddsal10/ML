import React, { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AlertTriangle, ArrowLeft, CheckCircle2, Play, RefreshCw } from "lucide-react";
import { endpoints, errorMessage } from "../api/client";
import { Badge, CATEGORY_LABELS, Card, ErrorState, LoadingCard, PageHead, Tabs, fmt, heat, useFetch } from "../components/ui.jsx";

const tooltipStyle = { background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10 };
const axis = { fill: "var(--text-3)", fontSize: 12 };
const label = (k) => CATEGORY_LABELS[k] || k.replaceAll("_", " ");

function ConfusionMatrix({ matrix, labels, short }) {
  return (
    <div className="table-wrap">
      <table className="cm">
        <thead>
          <tr><th />{labels.map((l) => <th key={l} className={short ? "" : "rot"}>{short ? l : label(l)}</th>)}</tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => {
            const total = row.reduce((a, b) => a + b, 0) || 1;
            return (
              <tr key={i}>
                <th style={{ textAlign: "right", whiteSpace: "nowrap" }}>{short ? labels[i] : label(labels[i])}</th>
                {row.map((v, j) => {
                  const share = v / total;
                  const style = i === j ? heat(share) : v ? { background: `color-mix(in srgb, var(--bad) ${Math.round(15 + 60 * share)}%, var(--surface))` } : { background: "var(--surface-2)" };
                  return <td key={j} style={style} title={`true ${labels[i]} → predicted ${labels[j]}: ${v}`}>{v || ""}</td>;
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="faint" style={{ marginTop: 6 }}>Rows = true class, columns = predicted. Diagonal = correct.</p>
    </div>
  );
}

function ComparisonChart({ rows, keys }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={rows} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#8882" vertical={false} />
        <XAxis dataKey="name" tick={axis} interval={0} />
        <YAxis domain={[0, 1]} tick={axis} />
        <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmt(v)} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {keys.map(([k, name, color]) => <Bar isAnimationActive={false} key={k} dataKey={k} name={name} fill={color} radius={[6, 6, 0, 0]} />)}
      </BarChart>
    </ResponsiveContainer>
  );
}

function ModelTable({ rows, cols, best, bestLabel = "selected" }) {
  return (
    <div className="table-wrap">
      <table className="table">
        <thead><tr><th>Algorithm</th>{cols.map(([, n]) => <th key={n} className="num">{n}</th>)}</tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name} className={r.name === best ? "best" : ""}>
              <td>{(KT_LABELS[r.name] || r.name).replaceAll("_", " ")} {r.name === best && <Badge tone="good">{bestLabel}</Badge>}</td>
              {cols.map(([k]) => <td key={k} className="num">{r[k] === undefined || r[k] === null ? "—" : Number.isInteger(r[k]) && r[k] > 1 ? r[k].toLocaleString() : typeof r[k] === "number" ? fmt(r[k]) : r[k]}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ---------- symbol classifier ---------- */
function RetrainPanel({ versions, onChanged }) {
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    if (!job || job.status !== "running") return;
    const t = setInterval(async () => {
      try {
        const res = await endpoints.labJob(job.id);
        setJob(res.data);
        if (res.data.status !== "running") onChanged();
      } catch (err) {
        setError(errorMessage(err));
      }
    }, 1500);
    return () => clearInterval(t);
  }, [job, onChanged]);

  const start = async () => {
    setError(null);
    try {
      const res = await endpoints.labRetrain();
      setJob({ id: res.data.job_id, status: "running", log: [] });
    } catch (err) {
      setError(errorMessage(err));
    }
  };
  const activate = async (v) => {
    try {
      await endpoints.labActivate("symbol_classifier", v);
      onChanged();
    } catch (err) {
      setError(errorMessage(err));
    }
  };
  const cmp = job?.result?.comparison;

  return (
    <Card title="Versions & active learning" subtitle="Students' symbol fixes are saved as labelled crops. Retraining adds them to the synthetic data and registers a new version for you to compare and activate.">
      <div className="stack">
        <div className="table-wrap">
          <table className="table">
            <thead><tr><th>Version</th><th>Created</th><th className="num">Held-out macro-F1</th><th className="num">Corrections</th><th>Notes</th><th /></tr></thead>
            <tbody>
              {versions.versions.map((v) => (
                <tr key={v.version} className={v.version === versions.active ? "best" : ""}>
                  <td><strong>{v.version}</strong> {v.version === versions.active && <Badge tone="good">active</Badge>}</td>
                  <td className="faint">{v.created_at ? new Date(v.created_at).toLocaleString() : "—"}</td>
                  <td className="num">{fmt(v.metrics?.held_out_macro_f1)}</td>
                  <td className="num">{v.metrics?.n_corrections ?? "—"}</td>
                  <td className="faint">{v.notes}</td>
                  <td>{v.version !== versions.active && <button className="btn sm secondary" onClick={() => activate(v.version)}>Activate</button>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="row">
          <button className="btn" onClick={start} disabled={job?.status === "running"}>
            {job?.status === "running" ? <span className="spinner" /> : <Play size={16} />} {job?.status === "running" ? "Retraining…" : "Retrain with corrections"}
          </button>
          <span className="faint">≈1–2 minutes on a laptop CPU</span>
        </div>
        {job?.log?.length > 0 && <div className="log">{job.log.map((l, i) => <div key={i}>› {l}</div>)}</div>}
        {cmp && (
          <div className="alert info">
            <CheckCircle2 size={18} />
            <div className="stack" style={{ gap: 4 }}>
              <strong>New version {job.result.version} ({job.result.selected_model})</strong>
              <span>Held-out fonts macro-F1: active {fmt(cmp.active?.macro_f1)} → new {fmt(cmp.new.macro_f1)}</span>
              {cmp.corrections_in_sample && <span>On {cmp.corrections_in_sample.n} corrected symbols: active {fmt(cmp.corrections_in_sample.active_accuracy)} → new {fmt(cmp.corrections_in_sample.new_accuracy)} (in-sample)</span>}
            </div>
          </div>
        )}
        {error && <ErrorState error={error} />}
      </div>
    </Card>
  );
}

function RealSymbolSection({ real }) {
  const DATA_LABELS = { synthetic_only: "Fonts only (synthetic)", real_only: "Real only (CROHME + HASYv2)", synthetic_plus_real: "Fonts + real" };
  const dataRows = Object.entries(real.training_data_comparison).map(([k, v]) => ({
    name: DATA_LABELS[k] || k, key: k, n_train: v.n_train, val: v.validation.selection_score,
    sym_acc: v.test_symbols.accuracy, sym_f1: v.test_symbols.macro_f1, expr: v.test_expressions.exact_match, cer: v.test_expressions.mean_cer,
  }));
  const algoRows = Object.entries(real.algorithm_comparison).map(([k, v]) => ({ name: k, val: v.validation.selection_score, acc: v.test_symbols.accuracy, f1: v.test_symbols.macro_f1, fit: v.fit_seconds }));
  const ds = real.datasets;
  const testN = Object.values(ds.crohme_test).reduce((a, d) => a + d.symbols, 0);
  return (
    <>
      <Card title="Real handwriting benchmark" subtitle={real.protocol} actions={<Badge tone="good">real data</Badge>}>
        <div className="grid grid-4" style={{ marginBottom: 16 }}>
          <div className="stat"><span className="stat-value">{ds.crohme_train.symbols.toLocaleString()}</span><span className="stat-label">CROHME train symbols ({ds.crohme_train.aligned} aligned expressions)</span></div>
          <div className="stat"><span className="stat-value">{ds.hasy.symbols.toLocaleString()}</span><span className="stat-label">HASYv2 symbols ({ds.hasy.writers} writers)</span></div>
          <div className="stat"><span className="stat-value">{testN.toLocaleString()}</span><span className="stat-label">CROHME 2014/16/19 test symbols</span></div>
          <div className="stat"><span className="stat-value">{Object.values(ds.crohme_test).reduce((a, d) => a + d.in_vocab_expressions, 0)}</span><span className="stat-label">whole test expressions</span></div>
        </div>
        <h3 style={{ marginBottom: 8 }}>Does real data matter? (same SVM, different training data)</h3>
        <ComparisonChart rows={dataRows} keys={[["sym_f1", "Test symbol macro-F1", "#6366f1"], ["expr", "Whole expressions exactly right", "#14b8a6"]]} />
        <ModelTable rows={dataRows} best={DATA_LABELS[real.selected_training_data]} cols={[["n_train", "Train n"], ["val", "Validation"], ["sym_acc", "Test acc."], ["sym_f1", "Test F1"], ["expr", "Expr. exact"], ["cer", "Expr. CER"]]} />
        <p className="faint" style={{ marginTop: 8 }}>A model trained only on fonts scores far lower on real handwriting than on held-out fonts — fonts are a weak proxy for writers.</p>
      </Card>
      <div className="split">
        <Card title="Algorithms on real handwriting" subtitle={`Training data: ${DATA_LABELS[real.selected_training_data]}`}>
          <ModelTable rows={algoRows} best={real.selected_model} cols={[["val", "Validation"], ["acc", "Test acc."], ["f1", "Test F1"], ["fit", "Fit s"]]} />
        </Card>
        <Card title="Confusion matrix — real test handwriting" subtitle={`${real.selected_model} on CROHME 2014/2016/2019`}>
          <ConfusionMatrix matrix={real.confusion_matrix} labels={real.labels} short />
        </Card>
      </div>
    </>
  );
}

function SymbolDetail({ report, real, versions, reload }) {
  const rows = Object.entries(report.models).map(([name, m]) => ({ name, ...m }));
  const e2e = report.end_to_end_ocr;
  return (
    <>
      {real && <RealSymbolSection real={real} />}
      {real && <h2 style={{ marginTop: 6 }}>Synthetic-font experiments</h2>}
      <div className="split">
        <Card title="Algorithm comparison (synthetic fonts)" subtitle={report.split}>
          <ComparisonChart rows={rows} keys={[["accuracy", "Accuracy", "#a5b4fc"], ["macro_f1", "Macro-F1", "#6366f1"]]} />
          <ModelTable rows={rows} best={report.selected_model} cols={[["accuracy", "Acc."], ["macro_f1", "Macro-F1"], ["fit_seconds", "Fit s"], ["predict_ms_per_symbol", "ms/symbol"]]} />
        </Card>
        <Card title="End-to-end reading" subtitle="Segmentation + recognition on whole expressions in held-out fonts">
          <div className="grid grid-2" style={{ marginBottom: 14 }}>
            <div className="stat"><span className="stat-value">{fmt(e2e.mean_cer)}</span><span className="stat-label">Character error rate</span></div>
            <div className="stat"><span className="stat-value">{fmt(e2e.exact_match, 2)}</span><span className="stat-label">Exact-match rate ({e2e.n_pages} pages)</span></div>
          </div>
          <h3 style={{ marginBottom: 8 }}>Demo images</h3>
          <table className="table">
            <tbody>
              {e2e.demo_images.map((d) => (
                <tr key={d.sample}><td className="mono">{d.reference.replace("\n", " ⏎ ")}</td><td className="mono">{d.hypothesis.replace("\n", " ⏎ ")}</td><td className="num">{d.cer === 0 ? <Badge tone="good">exact</Badge> : fmt(d.cer)}</td></tr>
              ))}
            </tbody>
          </table>
          <p className="faint" style={{ marginTop: 8 }}>{e2e.note}</p>
        </Card>
      </div>
      <Card title="Confusion matrix" subtitle={`${report.selected_model} on ${report.n_test} symbols from unseen fonts: ${report.held_out_fonts.join(", ")}`}>
        <ConfusionMatrix matrix={report.confusion_matrix} labels={report.labels} short />
      </Card>
      {versions && <RetrainPanel versions={versions} onChanged={reload} />}
    </>
  );
}

/* ---------- mistake classifier ---------- */
function MistakeDetail({ report }) {
  const [which, setWhich] = useState("model");
  const rows = Object.entries(report.models).map(([name, m]) => ({ name, accuracy: m.test_accuracy, macro_f1: m.test_macro_f1, cv: m.cv_macro_f1_mean }));
  const perClass = Object.entries(report.per_class_f1).map(([k, v]) => ({ name: label(k), f1: v }));
  const imp = report.top_feature_importances.map(([n, v]) => ({ name: n, importance: v }));
  return (
    <>
      <div className="split">
        <Card title="Algorithms vs rule-based baseline" subtitle={report.split}>
          <ComparisonChart rows={rows.map((r) => ({ ...r, name: r.name.replace("_baseline", "").replaceAll("_", " ") }))} keys={[["accuracy", "Test accuracy", "#a5b4fc"], ["macro_f1", "Test macro-F1", "#6366f1"]]} />
          <ModelTable rows={rows} best={report.selected_model} cols={[["cv", "CV F1"], ["accuracy", "Test acc."], ["macro_f1", "Test F1"]]} />
        </Card>
        <Card title="Which features matter" subtitle="Random-forest impurity importance (top 15)">
          <ResponsiveContainer width="100%" height={420}>
            <BarChart data={imp} layout="vertical" margin={{ left: 60, right: 10 }}>
              <XAxis type="number" tick={axis} />
              <YAxis type="category" dataKey="name" width={170} tick={{ fill: "var(--text-2)", fontSize: 11.5 }} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmt(v)} />
              <Bar isAnimationActive={false} dataKey="importance" fill="#14b8a6" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
      <div className="split">
        <Card title="Confusion matrix" actions={<Tabs value={which} onChange={setWhich} tabs={[{ id: "model", label: report.selected_model.replaceAll("_", " ") }, { id: "rules", label: "Rule baseline" }]} />}>
          <ConfusionMatrix matrix={which === "model" ? report.confusion_matrix : report.rule_based_confusion_matrix} labels={report.labels} />
        </Card>
        <Card title="Per-class F1" subtitle={report.selected_model.replaceAll("_", " ")}>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={perClass} layout="vertical" margin={{ left: 60, right: 10 }}>
              <XAxis type="number" domain={[0, 1]} tick={axis} />
              <YAxis type="category" dataKey="name" width={160} tick={{ fill: "var(--text-2)", fontSize: 11.5 }} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmt(v)} />
              <Bar isAnimationActive={false} dataKey="f1" fill="#6366f1" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </>
  );
}

/* ---------- knowledge tracing ---------- */
const KT_LABELS = { ema_baseline: "EMA (recency)", bkt: "BKT", pfa: "PFA", pfa_irt: "PFA + IRT", pfa_irt_recency: "PFA + IRT + recency" };

function RealTracingSection({ real }) {
  const rows = Object.entries(real.models).map(([k, m]) => ({ name: KT_LABELS[k] || k, key: k, ...m }));
  const best = Object.entries(real.models).sort((a, b) => b[1].auc - a[1].auc)[0][0];
  return (
    <>
      <Card title="Real students: ASSISTments 2009" subtitle={`${real.n_attempts.toLocaleString()} answers · ${real.n_learners.toLocaleString()} students · ${real.n_skills} skills · ${real.split}`} actions={<Badge tone="good">real data</Badge>}>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={rows} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#8882" vertical={false} />
            <XAxis dataKey="name" tick={axis} interval={0} />
            <YAxis domain={[0.5, 0.85]} ticks={[0.5, 0.6, 0.7, 0.8]} tick={axis} />
            <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmt(v)} />
            <Bar isAnimationActive={false} dataKey="auc" name="AUC" fill="#6366f1" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <ModelTable rows={rows.map((r) => ({ ...r, name: r.key }))} best={best} bestLabel="best" cols={[["auc", "AUC"], ["rmse", "RMSE"], ["log_loss", "Log-loss"], ["n", "Test answers"]]} />
        <p className="faint" style={{ marginTop: 8 }}>{real.note} Rasch regulariser chosen on validation students (λ = {real.rasch_l2_selected}). Y-axis starts at 0.5 (chance).</p>
      </Card>
      <Card title="Deployed BKT parameters (fitted on real students)" subtitle="StepWise skill ← matching ASSISTments skills">
        <div className="table-wrap">
          <table className="table">
            <thead><tr><th>Skill</th><th>ASSISTments skills</th><th className="num">Answers</th>{["L0", "T", "G", "S"].map((k) => <th key={k} className="num">{k}</th>)}</tr></thead>
            <tbody>
              {Object.entries(real.stepwise_bkt_params_from_real_data).map(([s, p]) => (
                <tr key={s}><td>{s}</td><td className="faint">{real.stepwise_skill_mapping[s].assistments_skills.join(", ")}</td>
                  <td className="num">{real.stepwise_skill_mapping[s].attempts.toLocaleString()}</td>
                  {["L0", "T", "G", "S"].map((k) => <td key={k} className="num">{fmt(p[k], 2)}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <h2 style={{ marginTop: 6 }}>Simulation studies</h2>
    </>
  );
}

function TracingDetail({ report, real }) {
  const kt = Object.entries(report.knowledge_tracing).map(([name, m]) => ({ name: name.replace("_baseline", "").toUpperCase(), ...m }));
  const policies = Object.entries(report.recommender_simulation.policies).map(([name, m]) => ({ name: name.replace("_", " "), mastered: m.skills_mastered_frac_mean, waste: m.practice_on_already_learned_frac }));
  return (
    <>
      {real && <RealTracingSection real={real} />}
      <div className="split">
        <Card title="Predicting the next answer (simulated learners)" subtitle={`${report.split} · ${report.data}`}>
          <ComparisonChart rows={kt} keys={[["auc", "AUC (higher is better)", "#6366f1"], ["rmse", "RMSE (lower is better)", "#f59e0b"]]} />
          <ModelTable rows={kt} best="BKT" cols={[["auc", "AUC"], ["rmse", "RMSE"], ["log_loss", "Log-loss"]]} />
        </Card>
        <Card title="Does adaptive practice help?" subtitle={report.recommender_simulation.setup}>
          <ComparisonChart rows={policies} keys={[["mastered", "Skills mastered", "#22c55e"], ["waste", "Practice on already-learned skills", "#ef4444"]]} />
          <p className="faint">Adaptive = practise the least-mastered skill (BKT) with items at ≈70% predicted success (PFA + IRT).</p>
        </Card>
      </div>
      <div className="split">
        <Card title="BKT parameter recovery" subtitle={`EM-fitted vs true simulator parameters · mean abs. error ${fmt(report.bkt_parameter_recovery_mae)}`}>
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th>Skill</th>{["L0", "T", "G", "S"].map((k) => <th key={k} className="num">{k} fit / true</th>)}</tr></thead>
              <tbody>
                {Object.entries(report.bkt_parameters).map(([s, p]) => (
                  <tr key={s}><td>{s}</td>{["L0", "T", "G", "S"].map((k) => <td key={k} className="num">{fmt(p.fitted[k], 2)} / {fmt(p.true[k], 2)}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
        <Card title="Item difficulty (Rasch IRT)">
          <div className="grid grid-3" style={{ marginBottom: 14 }}>
            <div className="stat"><span className="stat-value">{report.irt.n_items}</span><span className="stat-label">items</span></div>
            <div className="stat"><span className="stat-value">{fmt(report.irt.difficulty_corr_with_truth, 2)}</span><span className="stat-label">corr. with true difficulty</span></div>
            <div className="stat"><span className="stat-value">{fmt(report.irt.difficulty_model_cv_r2, 2)}</span><span className="stat-label">feature model CV R²</span></div>
          </div>
          <h3 style={{ marginBottom: 8 }}>What makes an item hard (ridge coefficients)</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={Object.entries(report.irt.difficulty_model_coefficients).map(([name, v]) => ({ name: name.replaceAll("_", " "), v }))} layout="vertical" margin={{ left: 40 }}>
              <XAxis type="number" tick={axis} />
              <YAxis type="category" dataKey="name" width={130} tick={{ fill: "var(--text-2)", fontSize: 11.5 }} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmt(v)} />
              <Bar isAnimationActive={false} dataKey="v" fill="#8b5cf6" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </>
  );
}

function ClusteringDetail({ report }) {
  return (
    <div className="split">
      <Card title="Quality on a simulated class" subtitle={report.data}>
        <div className="grid grid-2">
          {Object.entries(report.metrics).map(([k, v]) => (
            <div key={k} className="stat"><span className="stat-value">{fmt(v, 2)}</span><span className="stat-label">{k.replaceAll("_", " ")}</span></div>
          ))}
        </div>
      </Card>
      <Card title={`Groups found (k = ${report.k_selected})`}>
        <ul className="evidence">{Object.entries(report.cluster_names).map(([k, n]) => <li key={k}><Badge>{k}</Badge>{n}</li>)}</ul>
        <p className="faint" style={{ marginTop: 10 }}>Live groups for your students are on the <Link to="/teacher">Class overview</Link>.</p>
      </Card>
    </div>
  );
}

export default function LabModelPage() {
  const { name } = useParams();
  const { data, error, loading, reload } = useFetch(() => endpoints.labModel(name), [name]);
  if (loading && !data) return <LoadingCard lines={8} />;
  if (error) return <ErrorState error={error} onRetry={reload} />;
  const r = data.report;
  return (
    <>
      <Link to="/lab" className="row small" style={{ fontWeight: 600 }}><ArrowLeft size={15} /> All models</Link>
      <PageHead title={data.title} description={`${data.stage} · ${data.kind}`}>
        <button className="btn secondary" onClick={reload}><RefreshCw size={15} /> Refresh</button>
      </PageHead>

      <Card>
        <dl className="kv">
          <dt>Input</dt><dd>{data.input}</dd>
          <dt>Output</dt><dd>{data.output}</dd>
          <dt>Features</dt><dd>{data.features}</dd>
          <dt>Training data</dt><dd>{data.data}</dd>
          <dt>Evaluation</dt><dd>{data.split}</dd>
          <dt>Algorithms</dt><dd className="chips">{data.algorithms.map((a) => <Badge key={a}>{a}</Badge>)}</dd>
        </dl>
        <div className="alert warn" style={{ marginTop: 14 }}><AlertTriangle size={16} />{data.limitations}</div>
      </Card>

      {!r ? <ErrorState error="This model has no report yet — run python -m app.ml.train in backend/." /> : (
        <>
          {name === "symbol_classifier" && <SymbolDetail report={r} real={data.real_report} versions={data.versions} reload={reload} />}
          {name === "mistake_classifier" && <MistakeDetail report={r} />}
          {name === "knowledge_tracing" && <TracingDetail report={r} real={data.real_report} />}
          {name === "student_clustering" && <ClusteringDetail report={r} />}
        </>
      )}
    </>
  );
}
