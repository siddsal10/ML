import React, { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { ArrowRight, CheckCircle2, Info, PartyPopper, RefreshCw, Target, XCircle } from "lucide-react";
import { endpoints, errorMessage } from "../api/client";
import { useStudent } from "../api/StudentContext.jsx";
import { Badge, Card, Empty, ErrorState, LoadingCard, Meter, PageHead, pct, toneFor, useFetch } from "../components/ui.jsx";

const REASONS = {
  latest_mistake: "Picked because of the mistake in your latest work.",
  lowest_mastery: "Picked because it's your least-mastered skill.",
  requested: "The skill you chose.",
};

export default function PracticePage() {
  const { selectedStudentId } = useStudent();
  const [params, setParams] = useSearchParams();
  const skillParam = params.get("skill");
  const skills = useFetch(() => endpoints.listSkills(), []);
  const set = useFetch(() => (selectedStudentId ? endpoints.nextPractice(selectedStudentId, skillParam) : Promise.resolve(null)), [selectedStudentId, skillParam]);

  const [index, setIndex] = useState(0);
  const [answer, setAnswer] = useState("");
  const [result, setResult] = useState(null);
  const [results, setResults] = useState([]);
  const [mastery, setMastery] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  useEffect(() => {
    setIndex(0); setResult(null); setResults([]); setAnswer("");
    setMastery(set.data?.mastery ?? null);
  }, [set.data]);
  useEffect(() => { inputRef.current?.focus(); }, [index, set.data]);

  if (!selectedStudentId) return <Empty title="Choose a student">Pick or create a student in the top bar to start practising.</Empty>;
  if (set.loading && !set.data) return <LoadingCard lines={5} />;
  if (set.error) return <ErrorState error={set.error} onRetry={set.reload} />;

  const items = set.data?.items || [];
  const item = items[index];
  const done = index >= items.length;

  const submit = async (e) => {
    e.preventDefault();
    if (!answer.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const res = await endpoints.answerPractice({ student_id: selectedStudentId, item_id: item.id, answer: answer.trim() });
      setResult(res.data);
      setResults((r) => [...r, res.data.correct]);
      setMastery(res.data.mastery_after);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };
  const next = () => { setIndex((i) => i + 1); setResult(null); setAnswer(""); };

  return (
    <>
      <PageHead title="Practice" description="Questions are chosen by an IRT + knowledge-tracing model so each one is challenging but achievable (about 70% predicted success).">
        <select aria-label="Skill" value={skillParam || ""} onChange={(e) => setParams(e.target.value ? { skill: e.target.value } : {})} style={{ width: 260 }}>
          <option value="">Recommended skill</option>
          {(skills.data || []).map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
        </select>
      </PageHead>

      <div className="split">
        <Card>
          <div className="stack-lg">
            <div className="row between wrap">
              <Badge tone="accent" icon={Target}>{set.data.skill.name}</Badge>
              <div className="dots" aria-label="Progress">
                {items.map((_, i) => <span key={i} className={i < results.length ? (results[i] ? "good" : "bad") : i === index ? "done" : ""} />)}
              </div>
            </div>

            {!done ? (
              <form onSubmit={submit} className="stack">
                <div className="row between">
                  <span className="faint">Question {index + 1} of {items.length} · {item.level}</span>
                  <span className="faint">predicted success {pct(item.predicted_p)}</span>
                </div>
                <div className="question">{item.prompt}</div>
                <div className="answer-row">
                  <input ref={inputRef} value={answer} onChange={(e) => setAnswer(e.target.value)} placeholder="Your answer" disabled={!!result} aria-label="Answer" />
                  {!result && <button className="btn" disabled={busy || !answer.trim()}>{busy ? <span className="spinner" /> : null} Check</button>}
                </div>
                {error && <ErrorState error={error} />}
                {result && (
                  <div className="stack">
                    <div className={`alert ${result.correct ? "good" : "bad"}`}>
                      {result.correct ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
                      <div className="stack" style={{ gap: 4 }}>
                        <strong>{result.correct ? "Correct!" : `Not quite — the answer is ${result.expected_answer}`}</strong>
                        <span>{result.explanation}</span>
                      </div>
                    </div>
                    <button type="button" className="btn" onClick={next} style={{ alignSelf: "flex-start" }}>
                      {index + 1 < items.length ? "Next question" : "Finish"} <ArrowRight size={16} />
                    </button>
                  </div>
                )}
              </form>
            ) : (
              <div className="empty">
                <PartyPopper size={34} color="var(--accent)" />
                <strong>Set complete — {results.filter(Boolean).length} of {results.length} correct</strong>
                <div className="row">
                  <button className="btn" onClick={set.reload}><RefreshCw size={16} /> Next set</button>
                  <Link className="btn secondary" to="/progress">See progress</Link>
                </div>
              </div>
            )}
          </div>
        </Card>

        <div className="stack-lg">
          <Card title="Skill mastery" subtitle="Bayesian Knowledge Tracing: probability you've learned this skill">
            <div className="stack">
              <div className="row between"><span className="stat-value">{pct(mastery)}</span>{mastery >= 0.95 && <Badge tone="good">Mastered</Badge>}</div>
              <Meter value={mastery} tone={toneFor(mastery)} label="Mastery" />
              {result && (
                <span className="faint">
                  {pct(result.mastery_before)} → {pct(result.mastery_after)} after this answer (model expected {pct(result.predicted_p)} chance of success)
                </span>
              )}
            </div>
          </Card>
          <Card>
            <div className="row" style={{ alignItems: "flex-start" }}>
              <Info size={18} color="var(--accent)" style={{ flex: "none" }} />
              <span className="muted small">{REASONS[set.data.reason]} Items you've already answered correctly are skipped. Mastery counts as reached at 95%.</span>
            </div>
          </Card>
        </div>
      </div>
    </>
  );
}
