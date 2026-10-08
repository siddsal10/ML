import React, { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import { Camera, Eraser, ImageUp, Pen, RotateCcw, Trash2 } from "lucide-react";

/* ---------------- Drop zone ---------------- */
export function Dropzone({ file, onFile }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);
  const [preview, setPreview] = useState(null);
  useEffect(() => {
    if (!file) return setPreview(null);
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  return (
    <div
      className={`dropzone ${dragging ? "dragging" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => { e.preventDefault(); setDragging(false); e.dataTransfer.files[0] && onFile(e.dataTransfer.files[0]); }}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
      role="button"
      tabIndex={0}
      aria-label="Upload an image of handwritten work"
    >
      <input ref={inputRef} type="file" accept=".png,.jpg,.jpeg,.webp" hidden onChange={(e) => e.target.files[0] && onFile(e.target.files[0])} />
      {preview ? (
        <div className="stack" style={{ alignItems: "center" }}>
          <img src={preview} alt="Selected work" />
          <span className="faint">{file.name} · click or drop to replace</span>
        </div>
      ) : (
        <div className="stack" style={{ alignItems: "center", gap: 8 }}>
          <ImageUp size={34} className="muted" />
          <strong>Drop a photo of handwritten work</strong>
          <span className="faint">or click to browse — PNG, JPG or WEBP · one problem, steps on separate lines</span>
        </div>
      )}
    </div>
  );
}

/* ---------------- Drawing canvas ---------------- */
export const DrawCanvas = forwardRef(function DrawCanvas({ width = 900, height = 380, lineWidth = 7, onChange }, ref) {
  const canvasRef = useRef(null);
  const drawing = useRef(false);
  const last = useRef(null);
  const history = useRef([]);
  const [tool, setTool] = useState("pen");
  const [empty, setEmpty] = useState(true);

  const ctx = () => canvasRef.current.getContext("2d");
  const clear = useCallback(() => {
    const c = ctx();
    c.fillStyle = "#ffffff";
    c.fillRect(0, 0, width, height);
    history.current = [];
    setEmpty(true);
    onChange?.(true);
  }, [width, height, onChange]);

  useEffect(clear, [clear]);

  useImperativeHandle(ref, () => ({
    toBlob: () => new Promise((resolve) => canvasRef.current.toBlob(resolve, "image/png")),
    clear,
    isEmpty: () => empty,
  }));

  const point = (e) => {
    const r = canvasRef.current.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * width, ((e.clientY - r.top) / r.height) * height];
  };
  const down = (e) => {
    e.preventDefault();
    canvasRef.current.setPointerCapture(e.pointerId);
    history.current.push(ctx().getImageData(0, 0, width, height));
    if (history.current.length > 30) history.current.shift();
    drawing.current = true;
    last.current = point(e);
    move(e);
  };
  const move = (e) => {
    if (!drawing.current) return;
    const c = ctx();
    const [x, y] = point(e);
    c.strokeStyle = tool === "pen" ? "#111827" : "#ffffff";
    c.lineWidth = tool === "pen" ? lineWidth : lineWidth * 4;
    c.lineCap = "round";
    c.lineJoin = "round";
    c.beginPath();
    c.moveTo(...last.current);
    c.lineTo(x, y);
    c.stroke();
    last.current = [x, y];
    if (empty) { setEmpty(false); onChange?.(false); }
  };
  const up = () => { drawing.current = false; };
  const undo = () => {
    const prev = history.current.pop();
    if (prev) ctx().putImageData(prev, 0, 0);
    if (!history.current.length) { setEmpty(true); onChange?.(true); }
  };

  return (
    <div className="stack">
      <div className="canvas-tools">
        <div className="tabs">
          <button type="button" className={`tab ${tool === "pen" ? "active" : ""}`} onClick={() => setTool("pen")}><Pen size={15} /> Pen</button>
          <button type="button" className={`tab ${tool === "eraser" ? "active" : ""}`} onClick={() => setTool("eraser")}><Eraser size={15} /> Eraser</button>
        </div>
        <button type="button" className="btn sm secondary" onClick={undo}><RotateCcw size={15} /> Undo</button>
        <button type="button" className="btn sm secondary" onClick={clear}><Trash2 size={15} /> Clear</button>
        <span className="faint">Write each step on its own line; leave a small gap between symbols.</span>
      </div>
      <div className="canvas-wrap">
        <canvas ref={canvasRef} width={width} height={height} onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerLeave={up} aria-label="Drawing area" />
      </div>
    </div>
  );
});

/* ---------------- Camera ---------------- */
export function CameraCapture({ onCapture }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [error, setError] = useState(null);
  const [shot, setShot] = useState(null);

  useEffect(() => {
    let cancelled = false;
    navigator.mediaDevices?.getUserMedia({ video: { facingMode: "environment" } })
      .then((stream) => {
        if (cancelled) return stream.getTracks().forEach((t) => t.stop());
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
      })
      .catch(() => setError("Camera not available — check browser permissions, or use Upload instead."));
    if (!navigator.mediaDevices) setError("This browser does not support camera capture.");
    return () => { cancelled = true; streamRef.current?.getTracks().forEach((t) => t.stop()); };
  }, []);

  const capture = () => {
    const v = videoRef.current;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d").drawImage(v, 0, 0);
    c.toBlob((blob) => {
      const file = new File([blob], "camera.png", { type: "image/png" });
      setShot(URL.createObjectURL(blob));
      onCapture(file);
    }, "image/png");
  };

  if (error) return <div className="alert warn">{error}</div>;
  return (
    <div className="stack">
      <div className="video-wrap">{shot ? <img src={shot} alt="Captured" /> : <video ref={videoRef} autoPlay playsInline muted />}</div>
      <div className="row">
        {shot ? (
          <button type="button" className="btn secondary" onClick={() => { setShot(null); onCapture(null); }}><RotateCcw size={16} /> Retake</button>
        ) : (
          <button type="button" className="btn" onClick={capture}><Camera size={16} /> Capture</button>
        )}
      </div>
    </div>
  );
}
