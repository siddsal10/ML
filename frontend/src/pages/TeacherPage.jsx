import React, { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { Lightbulb, Users } from "lucide-react";
import { endpoints } from "../api/client";
import { useStudent } from "../api/StudentContext.jsx";
import { useNavigate } from "react-router-dom";
import { Badge, CATEGORY_LABELS, Card, CategoryBadge, Empty, ErrorState, LoadingCard, PageHead, Stat, fmt, heat, pct, useFetch } from "../components/ui.jsx";

const CLUSTER_COLORS = ["#6366f1", "#f59e0b", "#14b8a6", "#ec4899", "#8b5cf6", "#0ea5e9"];

export default function TeacherPage() {
  const { data, error, loading, reload } = useFetch(() => endpoints.classOverview(), []);
  const { setSelectedStudentId } = useStudent();
  const navigate = useNavigate();
  const [focus, setFocus] = useState(null);

  const byCluster = useMemo(() => {
    const groups = {};
    (data?.students || []).forEach((s) => { (groups[s.cluster] ||= []).push(s); });
    return groups;
  }, [data]);

  if (loading && !data) return <LoadingCard lines={8} />;
  if (error) return <ErrorState error={error} onRetry={reload} />;
  if (!data.students.length) return <Empty icon={Users} title="No students yet">Run <code>make seed-demo</code> to create a simulated class.</Empty>;

  const students = focus === null ? data.students : data.students.filter((s) => String(s.cluster) === String(focus));
  const classAvg = data.students.reduce((a, s) => a + Object.values(s.mastery).reduce((x, y) => x + y, 0) / 8, 0) / data.students.length;
  const skillAvg = data.skills.map((sk) => ({ ...sk, avg: data.students.reduce((a, s) => a + s.mastery[sk.id], 0) / data.students.length }));
  const weakestSkill = [...skillAvg].sort((a, b) => a.avg - b.avg)[0];
  const open = (id) => { setSelectedStudentId(id); navigate("/progress"); };

  return (
    <>
      <PageHead title="Class overview" description="Students are grouped automatically by their misconception profile (K-Means on mistake rates + BKT mastery). Use the groups to plan small-group teaching." />

      <div className="grid grid-4">
        <Stat label="Students" value={data.students.length} />
        <Stat label="Class mastery" value={pct(classAvg)} />
        <Stat label="Weakest skill" value={weakestSkill.short} hint={pct(weakestSkill.avg)} />
        <Stat label="Groups found" value={Object.keys(data.clusters).length} hint={data.cluster_metrics.kmeans_silhouette !== undefined ? `silhouette ${fmt(data.cluster_metrics.kmeans_silhouette, 2)}` : ""} />
      </div>

      <div className="grid grid-3">
        {Object.entries(data.clusters).map(([id, c]) => (
          <button key={id} className="card model-card" style={{ textAlign: "left", font: "inherit", cursor: "pointer", outline: String(focus) === id ? "2px solid var(--accent)" : "none" }}
                  onClick={() => setFocus(String(focus) === id ? null : id)} aria-pressed={String(focus) === id}>
            <div className="row between">
              <span className="row"><span style={{ width: 12, height: 12, borderRadius: 4, background: CLUSTER_COLORS[id % 6] }} /> <strong>{c.name}</strong></span>
              <Badge>{c.size} students</Badge>
            </div>
            <span className="faint">avg mastery {pct(c.avg_mastery)}</span>
            <div className="row small muted" style={{ alignItems: "flex-start" }}><Lightbulb size={15} color="var(--warn)" style={{ flex: "none", marginTop: 2 }} />{c.suggestion}</div>
          </button>
        ))}
      </div>

      <div className="split">
        <Card title="Misconception map" subtitle="PCA projection of each student's profile — nearby students struggle in similar ways">
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart margin={{ top: 10, right: 10, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#8882" />
              <XAxis type="number" dataKey="x" name="PC1" tick={{ fill: "var(--text-3)", fontSize: 11 }} />
              <YAxis type="number" dataKey="y" name="PC2" tick={{ fill: "var(--text-3)", fontSize: 11 }} />
              <ZAxis range={[90, 90]} />
              <Tooltip cursor={{ strokeDasharray: "3 3" }}
                       content={({ payload }) => payload?.[0] ? (
                         <div className="card tight small"><strong>{payload[0].payload.name}</strong><div className="faint">{data.clusters[payload[0].payload.cluster]?.name}</div></div>
                       ) : null} />
              {Object.entries(byCluster).map(([cid, members]) => (
                <Scatter key={cid} isAnimationActive={false} data={members} fill={CLUSTER_COLORS[cid % 6]} onClick={(p) => open(p.id)} style={{ cursor: "pointer" }}
                         opacity={focus === null || String(focus) === cid ? 1 : 0.2} />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Most common misconceptions">
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={data.misconceptions.map((m) => ({ name: CATEGORY_LABELS[m.category] || m.category, count: m.count }))} layout="vertical" margin={{ left: 30, right: 16 }}>
              <XAxis type="number" allowDecimals={false} tick={{ fill: "var(--text-3)", fontSize: 12 }} />
              <YAxis type="category" dataKey="name" width={150} tick={{ fill: "var(--text-2)", fontSize: 12 }} />
              <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10 }} />
              <Bar isAnimationActive={false} dataKey="count" fill="#ef4444" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <Card title="Mastery heatmap" subtitle={focus === null ? "All students · click a row to open their progress" : `Group: ${data.clusters[focus]?.name} · click a group card again to show everyone`}>
        <div className="table-wrap">
          <table className="heatmap">
            <thead>
              <tr>
                <th className="name">Student</th>
                {data.skills.map((s) => <th key={s.id} title={s.name}>{s.short}</th>)}
                <th>Top mistake</th>
              </tr>
            </thead>
            <tbody>
              {students.map((s) => (
                <tr key={s.id} onClick={() => open(s.id)} style={{ cursor: "pointer" }}>
                  <td className="name"><span className="row" style={{ gap: 8 }}><span style={{ width: 8, height: 8, borderRadius: 99, background: CLUSTER_COLORS[s.cluster % 6] }} />{s.name}</span></td>
                  {data.skills.map((sk) => <td key={sk.id} style={heat(s.mastery[sk.id])}>{Math.round(s.mastery[sk.id] * 100)}</td>)}
                  <td style={{ textAlign: "left" }}>{s.top_mistake ? <CategoryBadge category={s.top_mistake} /> : <span className="faint">—</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  );
}
