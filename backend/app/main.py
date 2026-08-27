from fastapi import FastAPI

app = FastAPI(title="VisionInspect AI Backend")


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "VisionInspect AI backend is running"}
