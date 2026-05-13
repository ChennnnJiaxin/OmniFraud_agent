from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.error_handlers import register_error_handlers
from api.routes.agent_routes import router as agent_router
from api.routes.case_routes import router as case_router
from api.routes.qa_routes import router as qa_router
from api.routes.risk_routes import router as risk_router
from api.routes.session_routes import router as session_router
from api.routes.sms_routes import router as sms_router

app = FastAPI(
    title="OmniFraud API",
    version="0.2.0",
    description="HTTP API facade for OmniFraud services",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_error_handlers(app)

app.include_router(sms_router)
app.include_router(qa_router)
app.include_router(case_router)
app.include_router(risk_router)
app.include_router(agent_router)
app.include_router(session_router)


@app.get("/health")
def health_check() -> dict[str, object]:
    return {
        "success": True,
        "service": "OmniFraud API",
        "status": "ok",
    }
