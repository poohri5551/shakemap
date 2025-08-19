from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from .logic import run_pipeline

app = FastAPI(title="Quake PGA Web", version="1.0.0")

# ถ้าต้องการเปิดใช้จากโดเมนอื่น ๆ
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ปรับให้เหมาะสมในโปรดักชัน
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.post("/api/run")
def api_run():
    try:
        data = run_pipeline()
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/health")
def health():
    return {"status": "ok"}