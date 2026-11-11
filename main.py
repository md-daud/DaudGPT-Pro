from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app = FastAPI(title="DaudGPT Pro", version="2.0.0")

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

@app.get("/api/status")
def status():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        r.raise_for_status()
        return {"status": "online", "models": [m.get("name", "") for m in r.json().get("models", [])]}
    except requests.RequestException:
        return {"status": "offline", "models": []}

@app.post("/chat")
def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    history_text = ""
    for item in request.history[-10:]:
        role = "User" if item.get("role") == "user" else "DaudGPT"
        content = str(item.get("content", ""))[:3000]
        if content:
            history_text += f"{role}: {content}\\n"
    prompt = ("You are DaudGPT, a helpful, clear AI assistant. Answer in the language used by the user. "
              "Use Markdown when useful. If asked for code, provide correct readable code.\\n\\n"
              f"Recent conversation:\\n{history_text}\\nUser: {message}\\nDaudGPT:")
    try:
        response = requests.post("http://localhost:11434/api/generate",
            json={"model": "qwen2.5:1.5b", "prompt": prompt, "stream": False,
                  "options": {"num_ctx": 4096}}, timeout=300)
        response.raise_for_status()
        return {"response": response.json().get("response", "").strip()}
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail="Ollama se connect nahi ho pa raha. Check karein Ollama chal raha hai aur qwen2.5:1.5b installed hai.") from exc

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")
