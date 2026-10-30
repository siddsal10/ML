import React, { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  AlertTriangle, ArrowRight, BrainCircuit, CheckCircle2, CircleHelp, Lightbulb, MousePointerClick, PencilLine, Timer, XCircle,
} from "lucide-react";
import { endpoints, errorMessage } from "../api/client";
import { Badge, Card, CategoryBadge, ErrorState, LoadingCard, Meter, PageHead, pct, toneFor, useFetch } from "../components/ui.jsx";

const SYMBOLS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "=", "x", "(", ")", "/"];
const STAGE_LABELS = { preprocess: "Clean", regions: "Regions", ocr: "Read (SVM)", verify: "Verify (SymPy)", diagnose: "Diagnose (RF)", feedback: "Feedback" };
const BOX_COLORS = { good: "var(--good)", warn: "var(--warn)", bad: "var(--bad)" };

function SymbolOverlay({ data, pending, selected, onSelect }) {
  const { image_width: W, image_height: H, symbols } = data;
  return (
    <div className="overlay-wrap">
      <img src={endpoints.originalImageUrl(data.submission_id)} alt="Your handwritten work" />
      {W && H && (
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-label="Recognised symbols">
          {symbols.map((s) => {
            const label = pending[s.idx] ?? s.corrected_label ?? s.predicted;
            const fixed = pending[s.idx] !== undefined || s.corrected_label;
            const tone = fixed ? "good" : toneFor(s.confidence);
            const pad = 4;
            return (
              <g key={s.idx} onClick={() => onSelect(s.idx)} role="button" tabIndex={0} aria-label={`Symbol ${label}, ${pct(s.confidence)} confident`}
                 onKeyDown={(e) => e.key === "Enter" && onSelect(s.idx)}>
                <rect className={`sym-box ${selected === s.idx ? "selected" : ""}`} x={s.x - pad} y={s.y - pad} width={s.w + 2 * pad} height={s.h + 2 * pad}
                      rx={6} stroke={BOX_COLORS[tone]} strokeDasharray={fixed ? "0" : tone === "good" ? "0" : "6 4"} />
                <text x={s.x - pad} y={s.y - pad - 6} fontSize={Math.max(14, H / 22)} fontWeight="700" fill={BOX_COLORS[tone]} fontFamily="JetBrains Mono, monospace">
                  {label}
                </text>
                <title>{`"${label}" · ${pct(s.confidence)} · alternatives: ${s.alternatives.map(([c, p]) => `${c} ${pct(p)}`).join(", ")}`}</title>
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}

function StepList({ steps, suspected }) {
  if (!steps?.length) return <p className="muted">No steps could be read.</p>;
  return (
    <div className="steps-list">
      {steps.map((s, i) => {
        const bad = s.raw_text === suspected || s.is_valid === false;
        const Icon = bad ? XCircle : s.kind === "unparseable" ? CircleHelp : CheckCircle2;
        const color = bad ? "var(--bad)" : s.kind === "unparseable" ? "var(--warn)" : "var(--good)";
        const note = bad ? "first wrong step" : s.kind === "unparseable" ? "couldn't parse" : s.is_valid === true ? "true" : "consistent";
        return (
          <div key={i} className={`step-line ${bad ? "bad" : ""}`}>
            <Icon size={18} color={color} className="icon" />
            <span>{s.raw_text}</span>
            <span className="note">{note}</span>
          </div>
        );
      })}
    </div>
  );
}

function RealSecondOpinion({ info }) {
  if (!info) return null;
  const slip = info.slip_probability;
  return (
    <Card title="Second opinion" subtitle={`Trained on real data · ${info.trained_on}`} actions={<Badge tone="good">real data</Badge>}>
      <div className="stack">
        <div className="row between small">
          <span>Looks like a <strong>{info.kind}</strong></span>
          <strong>{pct(slip)} slip</strong>
        </div>
        <Meter value={slip} tone={slip >= 0.5 ? "warn" : "bad"} label="Probability this is a careless slip" />
        <span className="faint">{slip >= 0.5 ? "Slips need checking habits (estimate, re-read, substitute back)." : "Understanding gaps need re-teaching the idea, not just more practice."}</span>
        <div className="stack" style={{ gap: 6 }}>
          {info.top.map((t) => (
            <div key={t.category} className="row between small"><span className="muted">{t.label}</span><span className="mono">{pct(t.probability)}</span></div>
          ))}
        </div>
      </div>
    </Card>
  );
}

function DiagnosisCard({ data }) {
  const evidence = data.diagnostic_evidence || [];
  const modelLine = evidence.find((e) => e.startsWith("Classifier"));
  const facts = evidence.filter((e) => e !== modelLine);
  const isML = data.mistake_source === "ml";
  return (
    <Card className="accent">
      <div className="stack">
        <div className="row between wrap">
          <span className="faint">Diagnosis</span>
          <Badge tone={isML ? "accent" : ""} icon={BrainCircuit}>{isML ? "ML classifier" : "Rule-based"}</Badge>
        </div>
        <div className="row wrap" style={{ gap: 12 }}>
          {data.correctness === true ? <CheckCircle2 size={28} color="var(--good)" /> : data.correctness === false ? <XCircle size={28} color="var(--bad)" /> : <CircleHelp size={28} color="var(--warn)" />}
          <div className="diag-title">{data.correctness === true ? "All steps check out" : <CategoryBadge category={data.mistake_category} />}</div>
        </div>
        <div className="stack" style={{ gap: 6 }}>
          <div className="row between small"><span className="muted">Model confidence</span><strong>{pct(data.confidence)}</strong></div>
          <Meter value={data.confidence} tone={toneFor(data.confidence)} label="Diagnosis confidence" />
          {modelLine && <span className="faint">{modelLine}</span>}
        </div>
        {facts.length > 0 && (
          <div className="stack" style={{ gap: 8 }}>
            <h3>Why we think this</h3>
            <ul className="evidence">
              {facts.map((e) => (
                <li key={e}><CheckCircle2 size={15} color="var(--accent)" style={{ flex: "none", marginTop: 2 }} />{e}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </Card>
  );
}

export default function AnalysisPage() {
  const { submissionId } = useParams();
  const { data, error, loading, reload, setData } = useFetch(() => endpoints.getAnalysis(submissionId), [submissionId]);
  const [selected, setSelected] = useState(null);
  const [pending, setPending] = useState({});
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState(null);
  const [editText, setEditText] = useState(null);

  const uncertain = useMemo(() => (data?.symbols || []).filter((s) => s.confidence < (data.symbol_check_below ?? 0.6) && !s.corrected_label), [data]);
  const awaitingCheck = data?.mistake_category === "transcription_uncertainty" && !data?.symbols?.some((s) => s.corrected_label) && uncertain.length > 0;
  const selectedSym = data?.symbols?.find((s) => s.idx === selected);

  if (loading && !data) return <LoadingCard lines={6} />;
  if (error) return <ErrorState error={error} onRetry={reload} />;

  const choose = (label) => {
    setPending((p) => {
      const next = { ...p };
      if (label === selectedSym.predicted && !selectedSym.corrected_label) delete next[selected];
      else next[selected] = label;
      return next;
    });
  };

  const applyCorrections = async () => {
    setSaving(true);
    setSaveError(null);
    try {
      // With no pending fixes this confirms the reading as it is.
      const corrections = Object.entries(pending).map(([idx, label]) => ({ idx: Number(idx), label }));
      const res = await endpoints.correctSymbols(submissionId, corrections);
      setData(res.data);
      setPending({});
      setSelected(null);
    } catch (err) {
      setSaveError(errorMessage(err));
    } finally {
      setSaving(false);
    }
  };

  const applyText = async () => {
    setSaving(true);
    setSaveError(null);
    try {
      const res = await endpoints.reanalyzeSubmission(submissionId, editText);
      setData({ ...res.data, symbols: data.symbols, image_width: data.image_width, image_height: data.image_height });
      setEditText(null);
    } catch (err) {
      setSaveError(errorMessage(err));
    } finally {
      setSaving(false);
    }
  };

  const pendingCount = Object.keys(pending).length;
  const timings = Object.entries(data.stage_timings || {});

  return (
    <>
      <PageHead title={data.original_question || "Your work"} description={data.expected_answer ? `Answer key: ${data.expected_answer}` : "No answer key given — StepWise checks that every step is consistent."}>
        <Link className="btn" to={`/practice${data.skill ? `?skill=${data.skill}` : ""}`}>
          Practise this skill <ArrowRight size={16} />
        </Link>
      </PageHead>

      {data.is_demo_mode && <div className="alert warn"><AlertTriangle size={18} />Demo provider was used — this transcription is a fixture, not model output.</div>}
      {awaitingCheck ? (
        <div className="alert warn"><MousePointerClick size={18} />Before I diagnose, please check the {uncertain.length} highlighted symbol{uncertain.length > 1 ? "s" : ""} — so a misread isn’t mistaken for your error. Tap a box to fix it, or confirm the reading below.</div>
      ) : data.ocr_confidence < 0.75 && data.symbols?.length > 0 && (
        <div className="alert info"><MousePointerClick size={18} />I may have misread some of your writing. Tap any box on the image to fix a symbol — your fix also teaches the model.</div>
      )}

      <div className="split">
        <div className="stack-lg">
          <Card
            title="What I read"
            subtitle={data.symbols?.length ? `${data.symbols.length} symbols · OCR confidence ${pct(data.ocr_confidence)} · click a box to correct it` : `OCR confidence ${pct(data.ocr_confidence)}`}
            actions={<button className="btn sm ghost" onClick={() => setEditText(editText === null ? data.extracted_text : null)}><PencilLine size={15} /> Edit text</button>}
          >
            <div className="stack">
              <SymbolOverlay data={data} pending={pending} selected={selected} onSelect={setSelected} />
              <div className="legend">
                <span><i style={{ background: "var(--good)" }} />confident (≥80%)</span>
                <span><i style={{ background: "var(--warn)" }} />unsure</span>
                <span><i style={{ background: "var(--bad)" }} />likely wrong (&lt;50%)</span>
                {uncertain.length > 0 && <span>· {uncertain.length} to check</span>}
              </div>

              {selectedSym && (
                <div className="card tight stack" style={{ boxShadow: "none" }}>
                  <div className="row between">
                    <strong>Symbol #{selectedSym.idx + 1}: the model read “{selectedSym.predicted}” ({pct(selectedSym.confidence)})</strong>
                    <button className="btn sm ghost" onClick={() => setSelected(null)}>Close</button>
                  </div>
                  <span className="faint">Runner-ups: {selectedSym.alternatives.map(([c, p]) => `“${c}” ${pct(p)}`).join(" · ")}. What did you write?</span>
                  <div className="picker">
                    {SYMBOLS.map((c) => (
                      <button key={c} className={(pending[selected] ?? selectedSym.corrected_label ?? selectedSym.predicted) === c ? "current" : ""} onClick={() => choose(c)}>{c}</button>
                    ))}
                  </div>
                </div>
              )}

              {awaitingCheck && pendingCount === 0 && (
                <div className="row wrap">
                  <button className="btn" onClick={applyCorrections} disabled={saving}>
                    {saving ? <span className="spinner" /> : <CheckCircle2 size={16} />} Reading is right — check my work
                  </button>
                </div>
              )}

              {pendingCount > 0 && (
                <div className="row wrap">
                  <button className="btn" onClick={applyCorrections} disabled={saving}>
                    {saving ? <span className="spinner" /> : <CheckCircle2 size={16} />} Re-check with {pendingCount} fix{pendingCount > 1 ? "es" : ""}
                  </button>
                  <button className="btn ghost" onClick={() => setPending({})}>Discard</button>
                  <span className="faint">Fixed symbols are saved as training data for the next model version.</span>
                </div>
              )}

              {editText !== null && (
                <div className="stack">
                  <textarea rows={4} value={editText} onChange={(e) => setEditText(e.target.value)} aria-label="Transcription" />
                  <div className="row">
                    <button className="btn" onClick={applyText} disabled={saving || !editText.trim()}>Re-check text</button>
                    <button className="btn ghost" onClick={() => setEditText(null)}>Cancel</button>
                  </div>
                </div>
              )}
              {saveError && <ErrorState error={saveError} />}
            </div>
          </Card>

          <Card title="Step-by-step check" subtitle="Each line verified exactly with SymPy">
            <StepList steps={data.interpreted_steps} suspected={data.suspected_incorrect_step} />
          </Card>
        </div>

        <div className="stack-lg">
          <DiagnosisCard data={data} />
          <RealSecondOpinion info={data.real_error_type} />

          <Card title="What to do" actions={<Lightbulb size={18} color="var(--warn)" />}>
            <div className="stack">
              {data.feedback && <p>{data.feedback}</p>}
              {data.hint && <div className="alert info"><Lightbulb size={16} /> {data.hint}</div>}
              {data.prerequisite_concept && <span className="faint">Concept: {data.prerequisite_concept}</span>}
              {data.corrected_solution && (
                <details>
                  <summary>Show a worked solution</summary>
                  <pre className="solution">{data.corrected_solution}</pre>
                </details>
              )}
            </div>
          </Card>

          {data.warnings?.length > 0 && (
            <Card title="Notes">
              <ul className="evidence">{data.warnings.map((w) => <li key={w}><AlertTriangle size={15} color="var(--warn)" style={{ flex: "none", marginTop: 2 }} />{w}</li>)}</ul>
            </Card>
          )}

          {timings.length > 0 && (
            <Card title="Pipeline timing" actions={<Timer size={16} className="muted" />}>
              <div className="stack" style={{ gap: 8 }}>
                {timings.map(([k, ms]) => (
                  <div key={k} className="row between small"><span className="muted">{STAGE_LABELS[k] || k}</span><span className="mono">{ms.toFixed(1)} ms</span></div>
                ))}
                <span className="faint">Provider: {data.provider_used} · skill {data.skill || "—"}</span>
              </div>
            </Card>
          )}
        </div>
      </div>
    </>
  );
}
