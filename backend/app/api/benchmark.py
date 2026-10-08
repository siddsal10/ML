from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.database import get_db
from app.models.benchmark import BenchmarkResult, BenchmarkRun
from app.schemas.benchmark import BenchmarkResultOut, BenchmarkRunOut, BenchmarkRunRequest
from app.services.benchmark_service import export_results_csv, export_summary_json, run_benchmark

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])


def _default_manifest_path() -> str:
    settings = get_settings()
    backend_root = Path(settings.UPLOAD_DIR).parents[1]  # backend/
    return str((backend_root.parent / "evaluation" / "manifests" / "demo_manifest.jsonl").resolve())


@router.post("/run", response_model=BenchmarkRunOut)
def run_benchmark_endpoint(payload: BenchmarkRunRequest, db: Session = Depends(get_db)) -> BenchmarkRunOut:
    manifest_path = payload.manifest_path or _default_manifest_path()
    if not Path(manifest_path).exists():
        raise HTTPException(status_code=404, detail=f"Manifest not found: {manifest_path}")

    outcome = run_benchmark(manifest_path, payload.providers)

    run = BenchmarkRun(
        manifest_path=manifest_path,
        providers=payload.providers or list(outcome["summary"].keys()),
        status="completed",
        summary=outcome["summary"],
    )
    db.add(run)
    db.flush()

    for row in outcome["results"]:
        db.add(
            BenchmarkResult(
                run_id=run.id,
                provider=row["provider"],
                sample_id=row["sample_id"],
                available=row["available"],
                predicted_text=row.get("predicted_text"),
                reference_text=row.get("reference_text"),
                cer=row.get("cer"),
                wer=row.get("wer"),
                exact_match=row.get("exact_match"),
                normalized_edit_distance=row.get("normalized_edit_distance"),
                inference_time_ms=row.get("inference_time_ms"),
                diagnosis_correct=row.get("diagnosis_correct"),
                predicted_mistake_category=row.get("predicted_mistake_category"),
                reference_mistake_category=row.get("reference_mistake_category"),
            )
        )
    db.commit()
    db.refresh(run)

    # Also export flat files for the "research and evaluation" workflow.
    results_dir = Path(_default_manifest_path()).parents[1] / "results"
    export_results_csv(outcome["results"], results_dir / f"{run.id}.csv")
    export_summary_json(outcome["summary"], results_dir / f"{run.id}_summary.json")

    return _to_out(run)


@router.get("/{run_id}", response_model=BenchmarkRunOut)
def get_benchmark_run(run_id: str, db: Session = Depends(get_db)) -> BenchmarkRunOut:
    run = db.get(BenchmarkRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    return _to_out(run)


def _to_out(run: BenchmarkRun) -> BenchmarkRunOut:
    return BenchmarkRunOut(
        id=run.id,
        manifest_path=run.manifest_path,
        providers=run.providers,
        status=run.status,
        created_at=run.created_at,
        summary=run.summary,
        results=[BenchmarkResultOut(**{
            "provider": r.provider,
            "sample_id": r.sample_id,
            "available": r.available,
            "predicted_text": r.predicted_text,
            "reference_text": r.reference_text,
            "cer": r.cer,
            "wer": r.wer,
            "exact_match": r.exact_match,
            "normalized_edit_distance": r.normalized_edit_distance,
            "inference_time_ms": r.inference_time_ms,
            "diagnosis_correct": r.diagnosis_correct,
        }) for r in run.results],
    )
