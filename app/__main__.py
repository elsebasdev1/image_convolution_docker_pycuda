"""Arranque: `python -m app` (sin recarga automática ni depurador)."""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        workers=1,  # un único proceso = un único contexto CUDA
        reload=False,
        access_log=False,
        log_level="info",
    )
