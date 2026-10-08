import React, { useRef, useState } from "react";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Eye, Sparkles } from "lucide-react";
import { endpoints, errorMessage } from "../api/client";
import { DrawCanvas } from "../components/capture.jsx";
import { Badge, Card, Empty, ErrorState, PageHead, pct } from "../components/ui.jsx";

export default function PlaygroundPage() {
  const canvasRef = useRef(null);
  const [empty, setEmpty] = useState(true);
  const [result, setResult] = useState(null);
  const [selected, setSelected] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const predict = async () => {
    setBusy(true);
    setError(null);
    try {
      const blob = await canvasRef.current.toBlob();
      const fd = new FormData();
      fd.append("image", new File([blob], "draw.png", { type: "image/png" }));
      const res = await endpoints.labPredict(fd);
      setResult(res.data);
      setSelected(0);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const sym = result?.symbols?.[selected];
  return (
    <>
      <PageHead title="Symbol playground" description="Write symbols or a whole expression. The page is segmented with OpenCV, each symbol is turned into HOG + zoning features, and the active SVM returns class probabilities." />
      <div className="split">
        <Card title="Draw" actions={<button className="btn" onClick={predict} disabled={busy || empty}>{busy ? <span className="spinner" /> : <Sparkles size={16} />} Recognise</button>}>
          <DrawCanvas ref={canvasRef} width={900} height={300} onChange={setEmpty} />
          {error && <div style={{ marginTop: 12 }}><ErrorState error={error} /></div>}
        </Card>

        <div className="stack-lg">
          <Card title="Result" subtitle={result ? `model: ${result.model}` : "Draw something and press Recognise"}>
            {result ? (
              <div className="stack">
                <div className="question" style={{ padding: "4px 0" }}>{result.text || "—"}</div>
                <div className="chips">
                  {result.symbols.map((s, i) => (
                    <button key={i} className={`btn sm ${i === selected ? "" : "secondary"}`} onClick={() => setSelected(i)}>
                      <span className="mono">{s.top[0][0]}</span> {pct(s.top[0][1])}
                    </button>
                  ))}
                </div>
              </div>
            ) : <Empty icon={Eye} title="Nothing yet" />}
          </Card>

          {sym && (
            <Card title={`Symbol ${selected + 1}`} subtitle="What the classifier sees (28×28 normalised glyph) and its top-5 probabilities">
              <div className="row" style={{ alignItems: "flex-start", gap: 18 }}>
                <img src={sym.model_view} alt="Normalised glyph" width={112} height={112} className="glyph-grid" />
                <div style={{ flex: 1 }}>
                  <ResponsiveContainer width="100%" height={170}>
                    <BarChart data={sym.top.map(([c, p]) => ({ c: `“${c}”`, p }))} layout="vertical" margin={{ left: 0, right: 10 }}>
                      <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => pct(v)} tick={{ fill: "var(--text-3)", fontSize: 11 }} />
                      <YAxis type="category" dataKey="c" width={40} tick={{ fill: "var(--text)", fontSize: 13 }} />
                      <Tooltip formatter={(v) => pct(v, 1)} contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10 }} />
                      <Bar isAnimationActive={false} dataKey="p" fill="#6366f1" radius={[0, 6, 6, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
              {sym.top[0][1] < 0.5 && <div style={{ marginTop: 10 }}><Badge tone="warn">Low confidence — the app would ask the student to confirm this symbol</Badge></div>}
            </Card>
          )}
        </div>
      </div>
    </>
  );
}
