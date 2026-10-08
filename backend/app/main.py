from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import benchmark, health, lab, practice, progress, providers, quizzes, students, submissions, teacher
from app.core.config import get_settings
from app.core.logging_config import configure_logging, get_logger
from app.db.database import init_db

configure_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s (env=%s)", settings.APP_NAME, settings.ENV)
    init_db()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered multimodal personal tutor for handwritten educational content.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(students.router)
app.include_router(submissions.router)
app.include_router(quizzes.router)
app.include_router(progress.router)
app.include_router(providers.router)
app.include_router(benchmark.router)
app.include_router(practice.router)
app.include_router(teacher.router)
app.include_router(lab.router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
