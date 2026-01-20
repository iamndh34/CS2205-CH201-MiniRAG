from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
from app.db.mongodb import db
from app.api.endpoints.accounting import router
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Kết nối DB
    db.client = AsyncIOMotorClient(settings.MONGO_URL)
    print("Connected to MongoDB")
    yield
    # Shutdown: Đóng kết nối
    db.client.close()
    print("Disconnected MongoDB")

app = FastAPI(title="UIT MiniRAG Accounting", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/ai-uit")

# Đường dẫn tuyệt đối đến thư mục templates
TEMPLATES_DIR = Path(__file__).resolve().parent / "app" / "templates"

# Serve static files (images, css, js)
app.mount("/static", StaticFiles(directory=TEMPLATES_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Giao diện so sánh MiniRAG vs Traditional RAG"""
    return (TEMPLATES_DIR / "compare.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
