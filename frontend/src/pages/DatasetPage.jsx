import React from "react";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { endpoints } from "../api/client";
import { Badge, Card, Empty, ErrorState, LoadingCard, PageHead, Stat, useFetch } from "../components/ui.jsx";

export default function DatasetPage() {
  const { data, error, loading, reload } = useFetch(() => endpoints.labDataset(), []);
  if (loading && !data) return <LoadingCard lines={8} />;
  if (error) return <ErrorState error={error} onRetry={reload} />;
  const corrections = Object.entries(data.corrections).map(([c, n]) => ({ c, n }));
  const totalCorr = corrections.reduce((a, b) => a + b.n, 0);

  return (
    <>
      <PageHead title="Training data" description="The symbol recognizer is trained on real handwriting (CROHME, HASYv2) plus glyphs rendered from system fonts and distorted to look hand-drawn, plus every correction students make in the app. Knowledge tracing is fitted on real student logs (ASSISTments). Datasets are downloaded from their public sources by scripts/download_datasets.py and never redistributed." />
      {data.real_datasets ? (
        <Card title="Real datasets" actions={<Badge tone="good">real data</Badge>}>
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th>Dataset</th><th>Used for</th><th className="num">Size used</th><th>Source / licence</th></tr></thead>
              <tbody>
                <tr><td><strong>CROHME</strong> (train)</td><td>Symbol recognizer training</td><td className="num">{data.real_datasets.crohme_train.symbols.toLocaleString()} symbols from {data.real_datasets.crohme_train.aligned.toLocaleString()} expressions</td><td className="faint">ICDAR CROHME competitions · HF mirror Neeze/CROHME-full · research use</td></tr>
                <tr><td><strong>CROHME</strong> 2014/16/19 test</td><td>Real-handwriting evaluation (unseen writers)</td><td className="num">{Object.values(data.real_datasets.crohme_test).reduce((a, d) => a + d.symbols, 0).toLocaleString()} symbols</td><td className="faint">official competition test sets</td></tr>
                <tr><td><strong>HASYv2</strong></td><td>Symbol recognizer training</td><td className="num">{data.real_datasets.hasy.symbols.toLocaleString()} symbols, {data.real_datasets.hasy.writers} writers</td><td className="faint">Thoma 2017 · Zenodo 259444 · ODbL</td></tr>
                {data.assistments && <tr><td><strong>ASSISTments 2009</strong></td><td>Knowledge tracing benchmark + BKT parameters</td><td className="num">{data.assistments.n_attempts.toLocaleString()} answers, {data.assistments.n_learners.toLocaleString()} students</td><td className="faint">Feng, Heffernan &amp; Koedinger 2009 · research use</td></tr>}
              </tbody>
            </table>
          </div>
        </Card>
      ) : (
        <div className="alert warn">Real datasets not downloaded yet — run <code>python scripts/download_datasets.py</code> then <code>python -m app.ml.train</code>.</div>
      )}
      <div className="grid grid-4">
        <Stat label="Symbol classes" value={data.classes.length} />
        <Stat label="Synthetic fonts" value={data.fonts.length} />
        <Stat label="Samples per font & class" value="30" hint="1 clean + 29 augmented" />
        <Stat label="Student corrections" value={totalCorr} hint="used on next retrain" />
      </div>
      <div className="split">
        <Card title="Synthetic glyph samples" subtitle={data.grid_note}>
          <div className="row" style={{ alignItems: "flex-start" }}>
            <div className="stack" style={{ gap: 0, paddingTop: 2 }}>
              {data.grid_rows.map((c) => <div key={c} className="mono faint" style={{ height: 40, lineHeight: "40px", width: 20 }}>{c}</div>)}
            </div>
            <img src={data.grid} alt="Grid of rendered and augmented training glyphs" className="glyph-grid" />
          </div>
        </Card>
        <div className="stack-lg">
          <Card title="Augmentation" subtitle="Applied randomly to every synthetic sample">
            <ul className="evidence">
              <li><Badge>affine</Badge>rotation ±12°, shear ±0.3, x/y scale 0.8–1.2</li>
              <li><Badge>elastic</Badge>smoothed random displacement field (Simard et al., 2003)</li>
              <li><Badge>pen</Badge>stroke thickening / thinning</li>
              <li><Badge>split</Badge>25% of fonts held out entirely for evaluation</li>
            </ul>
          </Card>
          <Card title="Fonts used">
            <div className="chips">{data.fonts.map((f) => <Badge key={f}>{f.replace(/\.(ttf|ttc)$/, "")}</Badge>)}</div>
          </Card>
          <Card title="Corrections collected" subtitle="Saved when a student fixes a misread symbol on the Analysis page">
            {corrections.length ? (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={corrections}>
                  <XAxis dataKey="c" tick={{ fill: "var(--text)", fontSize: 13 }} />
                  <YAxis allowDecimals={false} tick={{ fill: "var(--text-3)", fontSize: 12 }} />
                  <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10 }} />
                  <Bar isAnimationActive={false} dataKey="n" fill="#6366f1" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <Empty title="No corrections yet">Fix a symbol on any analysis to start the active-learning loop.</Empty>}
          </Card>
        </div>
      </div>
    </>
  );
}
