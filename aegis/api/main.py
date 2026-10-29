"""FastAPI Application Entrypoint for Aegis."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from aegis.api.routes import router as api_router

app = FastAPI(
    title="Aegis AI Orchestrator API",
    description="Multi-Agent AI Engineering & Research Orchestrator with Stateful LangGraph Workflows",
    version="1.0.0",
)

# Enable CORS for local Vite or cross-origin dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)

# Mount frontend static directory if present
frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("aegis.api.main:app", host="0.0.0.0", port=8000, reload=True)
