import React, { useState } from "react";
import { Play } from "lucide-react";
import { endpoints, errorMessage } from "../api/client";
import { Badge, Card, ErrorState, PageHead, fmt, useFetch } from "../components/ui.jsx";

const val = (v, d = 3) => (v === null || v === undefined ? "—" : typeof v === "number" ? fmt(v, d) : v);

export default function BenchmarkPage() {
  const providers = useFetch(() => endpoints.listProviders(), []);
  const [selected, setSelected] = useState(["ml"]);
  const [manifest, setManifest] = useState("");
  const [running, setRunning] = useState(false);
  const [run, setRun] = useState(null);
  const [error, setError] = useState(null);

  const toggle = (name) => setSelected((p) => (p.includes(name) ? p.filter((x) => x !== name) : [...p, name]));
  const submit = async (e) => {
    e.preventDefault();
    setRunning(true);
    setError(null);
    try {
      const res = await endpoints.runBenchmark({ manifest_path: manifest.trim() || null, providers: selected.length ? selected : null });
      setRun(res.data);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setRunning(false);
    }
  };

  return (
    <>
      <PageHead title="OCR benchmark" description="Compare reading providers on a manifest of (image, reference transcription, gold mistake label). Use it to compare the classical-ML recognizer against the optional deep-learning baselines (TrOCR, Donut) on real handwriting." />
      <Card title="Run">
        <form className="stack" onSubmit={submit}>
          <label className="field">
            Manifest path (blank = bundled demo manifest)
            <input placeholder="evaluation/manifests/demo_manifest.jsonl" value={manifest} onChange={(e) => setManifest(e.target.value)} />
          </label>
          <div className="row wrap">
            {(providers.data || []).map((p) => (
              <label key={p.name} className="badge" style={{ cursor: "pointer", padding: "6px 12px" }}>
                <input type="checkbox" checked={selected.includes(p.name)} onChange={() => toggle(p.name)} style={{ width: "auto" }} />
                {p.name} {!p.available && <span className="faint">(unavailable)</span>}
              </label>
            ))}
          </div>
          {error && <ErrorState error={error} />}
          <button className="btn" disabled={running} style={{ alignSelf: "flex-start" }}>{running ? <span className="spinner" /> : <Play size={16} />} Run benchmark</button>
        </form>
      </Card>

      {run && (
        <>
          <Card title="Summary">
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Provider</th><th>Status</th><th className="num">Samples</th><th className="num">CER</th><th className="num">WER</th><th className="num">Exact</th><th className="num">ms / image</th><th className="num">Diagnosis acc.</th><th className="num">Macro-F1</th></tr></thead>
                <tbody>
                  {Object.entries(run.summary).map(([p, s]) => (
                    <tr key={p}>
                      <td><strong>{p}</strong></td>
                      <td>{s.available ? <Badge tone="good">available</Badge> : <Badge tone="warn">unavailable</Badge>}</td>
                      <td className="num">{s.num_samples}</td>
                      <td className="num">{val(s.mean_cer)}</td>
                      <td className="num">{val(s.mean_wer)}</td>
                      <td className="num">{val(s.exact_match_accuracy, 2)}</td>
                      <td className="num">{val(s.mean_inference_time_ms, 1)}</td>
                      <td className="num">{val(s.diagnosis_accuracy, 2)}</td>
                      <td className="num">{val(s.mistake_category_macro_f1, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <Card title="Per sample">
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Provider</th><th>Sample</th><th>Predicted</th><th>Reference</th><th className="num">CER</th></tr></thead>
                <tbody>
                  {run.results.map((r, i) => (
                    <tr key={i}><td>{r.provider}</td><td>{r.sample_id}</td><td className="mono">{r.predicted_text ?? "—"}</td><td className="mono">{r.reference_text ?? "—"}</td><td className="num">{val(r.cer)}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </>
  );
}
