import React, { useMemo } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Award, PenLine, Target } from "lucide-react";
import { endpoints } from "../api/client";
import { useStudent } from "../api/StudentContext.jsx";
import { Badge, CATEGORY_LABELS, Card, CategoryBadge, Empty, ErrorState, LoadingCard, Meter, PageHead, pct, Stat, toneFor, useFetch } from "../components/ui.jsx";

const SERIES = ["#6366f1", "#14b8a6", "#f59e0b", "#ef4444", "#8b5cf6", "#0ea5e9", "#22c55e", "#ec4899"];

export default function ProgressPage() {
  const { selectedStudentId } = useStudent();
  const navigate = useNavigate();
  const skills = useFetch(() => (selectedStudentId ? endpoints.getSkills(selectedStudentId) : Promise.resolve(null)), [selectedStudentId]);
  const progress = useFetch(() => (selectedStudentId ? endpoints.getProgress(selectedStudentId) : Promise.resolve(null)), [selectedStudentId]);
  const history = useFetch(() => (selectedStudentId ? endpoints.getHistory(selectedStudentId) : Promise.resolve(null)), [selectedStudentId]);

  const timeline = useMemo(() => {
    const events = skills.data?.timeline || [];
    return events.map((e, i) => ({ step: i + 1, [e.skill]: e.mastery, skill: e.skill, correct: e.correct, source: e.source }));
  }, [skills.data]);
  const practisedSkills = useMemo(() => [...new Set((skills.data?.timeline || []).map((e) => e.skill))], [skills.data]);

  if (!selectedStudentId) return <Empty title="Choose a student">Pick a student in the top bar to see their progress.</Empty>;
  if (skills.loading && !skills.data) return <LoadingCard lines={6} />;
  if (skills.error) return <ErrorState error={skills.error} onRetry={skills.reload} />;

  const sk = skills.data.skills;
  const mastered = sk.filter((s) => s.mastered).length;
  const avg = sk.reduce((a, s) => a + s.mastery, 0) / sk.length;
  const weakest = [...sk].sort((a, b) => a.mastery - b.mastery)[0];
  const mistakes = (progress.data?.mistake_category_counts || []).filter((m) => !["correct_solution"].includes(m.category))
    .map((m) => ({ name: CATEGORY_LABELS[m.category] || m.category, count: m.count }));
  const subs = history.data?.submissions || [];
  const attempts = skills.data.timeline.length;

  return (
    <>
      <PageHead title="My progress" description="Mastery per skill is estimated with Bayesian Knowledge Tracing from every practice answer and every diagnosed piece of handwritten work.">
        <Link className="btn secondary" to="/"><PenLine size={16} /> New work</Link>
        <Link className="btn" to={`/practice?skill=${weakest.id}`}><Target size={16} /> Practise weakest skill</Link>
      </PageHead>

      <div className="grid grid-4">
        <Stat label="Average mastery" value={pct(avg)} />
        <Stat label="Skills mastered" value={`${mastered} / ${sk.length}`} hint="P(learned) ≥ 95%" />
        <Stat label="Observations" value={attempts} hint="practice + handwritten" />
        <Stat label="Focus next" value={weakest.short} hint={pct(weakest.mastery)} />
      </div>

      <div className="split">
        <Card title="Skills" subtitle={`Model: ${skills.data.model === "bkt" ? "BKT with EM-fitted parameters" : "BKT with default parameters (model not trained)"}`}>
          <div className="stack">
            {sk.map((s) => (
              <div key={s.id} className="stack" style={{ gap: 6 }}>
                <div className="row between small">
                  <span>{s.name}</span>
                  <span className="row">
                    {s.mastered && <Badge tone="good" icon={Award}>Mastered</Badge>}
                    <span className="faint">{s.attempts} obs.</span>
                    <strong style={{ minWidth: 42, textAlign: "right" }}>{pct(s.mastery)}</strong>
                  </span>
                </div>
                <Meter value={s.mastery} tone={toneFor(s.mastery)} label={s.name} />
              </div>
            ))}
          </div>
        </Card>

        <Card title="Mastery over time" subtitle="Each point is one observation of that skill">
          {timeline.length ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={timeline} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#8882" />
                <XAxis dataKey="step" tick={{ fill: "var(--text-3)", fontSize: 12 }} />
                <YAxis domain={[0, 1]} tickFormatter={(v) => `${v * 100}%`} tick={{ fill: "var(--text-3)", fontSize: 12 }} />
                <Tooltip formatter={(v) => pct(v)} contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10 }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {practisedSkills.map((s) => (
                  <Line isAnimationActive={false} key={s} type="linear" dataKey={s} name={sk.find((x) => x.id === s)?.short} stroke={SERIES[Number(s.slice(1)) - 1]} dot={{ r: 2 }} strokeWidth={2} connectNulls />
                ))}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <Empty title="No history yet">Analyse some work or do a practice set.</Empty>
          )}
        </Card>
      </div>

      <div className="split">
        <Card title="Recent handwritten work">
          {subs.length ? (
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>When</th><th>Question</th><th>Result</th></tr></thead>
                <tbody>
                  {subs.slice(0, 10).map((s) => (
                    <tr key={s.submission_id} className="clickable" onClick={() => navigate(`/analysis/${s.submission_id}`)}>
                      <td className="faint">{new Date(s.created_at).toLocaleDateString()}</td>
                      <td className="mono">{s.question}</td>
                      <td><CategoryBadge category={s.mistake_category} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty title="Nothing analysed yet" />
          )}
        </Card>
        <Card title="Mistake patterns" subtitle="From the ML diagnosis of your work">
          {mistakes.length ? (
            <ResponsiveContainer width="100%" height={Math.max(160, mistakes.length * 42)}>
              <BarChart data={mistakes} layout="vertical" margin={{ left: 20, right: 16 }}>
                <XAxis type="number" allowDecimals={false} tick={{ fill: "var(--text-3)", fontSize: 12 }} />
                <YAxis type="category" dataKey="name" width={150} tick={{ fill: "var(--text-2)", fontSize: 12 }} />
                <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10 }} />
                <Bar isAnimationActive={false} dataKey="count" fill="#6366f1" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <Empty title="No mistakes recorded" />
          )}
        </Card>
      </div>
    </>
  );
}
