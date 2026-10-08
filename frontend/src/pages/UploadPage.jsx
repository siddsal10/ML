import React, { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Camera, CheckCircle2, ImageUp, PenLine, ScanLine, Sparkles } from "lucide-react";
import { endpoints, errorMessage } from "../api/client";
import { useStudent } from "../api/StudentContext.jsx";
import { CameraCapture, DrawCanvas, Dropzone } from "../components/capture.jsx";
import { Card, ErrorState, PageHead, Tabs } from "../components/ui.jsx";

const STAGES = ["Clean image", "Find symbols", "Recognise (SVM)", "Check steps (SymPy)", "Diagnose (Random Forest)"];

const EXAMPLES = [
  { q: "Solve 2x + 4 = 10", a: "3" },
  { q: "-3 + 5", a: "2" },
  { q: "12 + 15", a: "27" },
];

export default function UploadPage() {
  const { selectedStudentId } = useStudent();
  const navigate = useNavigate();
  const [mode, setMode] = useState("upload");
  const [file, setFile] = useState(null);
  const [canvasEmpty, setCanvasEmpty] = useState(true);
  const canvasRef = useRef(null);
  const [question, setQuestion] = useState("");
  const [expected, setExpected] = useState("");
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState(0);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!busy) return;
    setStage(0);
    const t = setInterval(() => setStage((s) => Math.min(s + 1, STAGES.length - 1)), 450);
    return () => clearInterval(t);
  }, [busy]);

  const hasInput = mode === "draw" ? !canvasEmpty : !!file;

  const submit = async (e) => {
    e.preventDefault();
    setError(null);
    if (!selectedStudentId) return setError("Pick or create a student first (top bar).");
    let image = file;
    if (mode === "draw") {
      const blob = await canvasRef.current.toBlob();
      image = new File([blob], "drawing.png", { type: "image/png" });
    }
    if (!image) return setError("Add your work first.");

    const fd = new FormData();
    fd.append("student_id", selectedStudentId);
    fd.append("topic", /x/.test(question) ? "algebra" : "arithmetic");
    fd.append("original_question", question || "Handwritten work");
    if (expected.trim()) fd.append("expected_answer", expected.trim());
    fd.append("provider", "ml");
    fd.append("image", image);
    setBusy(true);
    try {
      const res = await endpoints.analyzeSubmission(fd);
      navigate(`/analysis/${res.data.submission_id}`);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHead
        title="Check my working"
        description="Upload, photograph or write your arithmetic or equation working. StepWise reads it, proves which step goes wrong, and names the misconception — then gives you practice for exactly that."
      />
      <form className="split" onSubmit={submit}>
        <Card
          title="Your work"
          actions={
            <Tabs
              value={mode}
              onChange={(m) => { setMode(m); setFile(null); }}
              tabs={[
                { id: "upload", label: "Upload", icon: ImageUp },
                { id: "camera", label: "Camera", icon: Camera },
                { id: "draw", label: "Write here", icon: PenLine },
              ]}
            />
          }
        >
          {mode === "upload" && <Dropzone file={file} onFile={setFile} />}
          {mode === "camera" && <CameraCapture onCapture={setFile} />}
          {mode === "draw" && <DrawCanvas ref={canvasRef} onChange={setCanvasEmpty} />}
        </Card>

        <div className="stack-lg">
          <Card title="About the problem" subtitle="Optional, but the answer key lets StepWise spot final-answer mistakes.">
            <div className="stack">
              <label className="field">
                Question
                <input placeholder="e.g. Solve 2x + 4 = 10" value={question} onChange={(e) => setQuestion(e.target.value)} />
              </label>
              <label className="field">
                Correct final answer
                <input placeholder="e.g. 3" value={expected} onChange={(e) => setExpected(e.target.value)} />
              </label>
              <div className="row wrap">
                <span className="faint">Try:</span>
                {EXAMPLES.map((ex) => (
                  <button type="button" key={ex.q} className="btn sm ghost" onClick={() => { setQuestion(ex.q); setExpected(ex.a); }}>
                    {ex.q}
                  </button>
                ))}
              </div>
              {error && <ErrorState error={error} />}
              <button className="btn lg" disabled={busy || !hasInput}>
                {busy ? <span className="spinner" /> : <Sparkles size={18} />} {busy ? "Analysing…" : "Analyse my work"}
              </button>
            </div>
          </Card>

          <Card title="What happens next" subtitle="Every stage is classical ML or exact maths — no neural networks, no LLMs.">
            <div className="stepper">
              {STAGES.map((s, i) => (
                <span key={s} className={`step-chip ${busy ? (i < stage ? "done" : i === stage ? "active" : "") : ""}`}>
                  {busy && i < stage ? <CheckCircle2 size={13} /> : <ScanLine size={13} />} {s}
                </span>
              ))}
            </div>
          </Card>
        </div>
      </form>
    </>
  );
}
