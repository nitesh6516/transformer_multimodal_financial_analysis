from __future__ import annotations

import json
import mimetypes
import os
import queue
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from flask import (
        Flask,
        Response,
        jsonify,
        make_response,
        render_template,
        request,
        send_from_directory,
        session,
        stream_with_context,
    )
    from werkzeug.exceptions import BadRequest
    from werkzeug.utils import secure_filename
except ImportError as exc:  # pragma: no cover - import guard is for first-run UX.
    raise SystemExit(
        "Missing Flask dependencies. Install them with:\n"
        "  py -m pip install -r requirements.txt\n\n"
        "Then run:\n"
        "  py multimodal.py"
    ) from exc

try:
    from flask_cors import CORS
except ImportError:  # pragma: no cover
    CORS = None

try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
except ImportError:  # pragma: no cover
    Limiter = None
    get_remote_address = None

try:
    from flask_socketio import SocketIO, emit, join_room, leave_room
except ImportError:  # pragma: no cover
    SocketIO = None
    emit = join_room = leave_room = None

try:
    from celery import Celery
except ImportError:  # pragma: no cover
    Celery = None


BASE_DIR = Path(__file__).resolve().parent


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_upload_dir() -> Path:
    configured = os.environ.get("AETHER_UPLOAD_DIR")
    if configured:
        return Path(configured)
    if env_flag("VERCEL"):
        return Path("/tmp/aether-uploads")
    return BASE_DIR / "static" / "uploads"


UPLOAD_DIR = resolve_upload_dir()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_MODALITIES = {"text", "image", "audio", "video"}
ALLOWED_UPLOADS = {
    "image": {"png", "jpg", "jpeg", "webp", "gif"},
    "audio": {"mp3", "wav", "ogg", "webm", "m4a"},
    "video": {"mp4", "webm", "mov", "m4v"},
}

@dataclass
class TaskResult:
    task_id: str
    status: str
    payload: dict[str, Any]


class LocalTaskQueue:
    """Tiny Celery-compatible fallback for local demos without a broker."""

    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max(2, os.cpu_count() or 2))
        self.results: dict[str, TaskResult] = {}

    def delay(self, fn: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> str:
        task_id = uuid.uuid4().hex
        self.results[task_id] = TaskResult(task_id, "queued", {})

        def run() -> None:
            self.results[task_id] = TaskResult(task_id, "running", {})
            try:
                payload = fn(*args, **kwargs)
                self.results[task_id] = TaskResult(task_id, "complete", payload)
            except Exception as exc:  # pragma: no cover - defensive worker boundary.
                self.results[task_id] = TaskResult(task_id, "error", {"error": str(exc)})

        self.executor.submit(run)
        return task_id

    def get(self, task_id: str) -> TaskResult | None:
        return self.results.get(task_id)


def make_celery(app: Flask) -> Any:
    if Celery is None or not app.config.get("CELERY_BROKER_URL"):
        return None
    celery = Celery(
        app.import_name,
        broker=app.config["CELERY_BROKER_URL"],
        backend=app.config["CELERY_RESULT_BACKEND"],
    )
    celery.conf.update(app.config)
    return celery


def infer_modality_from_file(filename: str, content_type: str | None) -> str:
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    for modality, extensions in ALLOWED_UPLOADS.items():
        if extension in extensions:
            return modality
    if content_type:
        return content_type.split("/", 1)[0]
    return "image"


def validate_modality(value: str | None) -> str:
    modality = (value or "").strip().lower()
    if modality not in ALLOWED_MODALITIES:
        raise BadRequest(f"type must be one of: {', '.join(sorted(ALLOWED_MODALITIES))}")
    return modality


def validate_upload(filename: str, modality: str) -> None:
    if "." not in filename:
        raise BadRequest("Upload must include a file extension.")
    extension = filename.rsplit(".", 1)[-1].lower()
    if extension not in ALLOWED_UPLOADS.get(modality, set()):
        allowed = ", ".join(sorted(ALLOWED_UPLOADS[modality]))
        raise BadRequest(f"{modality} uploads must be one of: {allowed}")


def synthesize_response(modality: str, payload: Any) -> dict[str, Any]:
    prompt = payload if isinstance(payload, str) else json.dumps(payload, default=str)[:280]
    prompt = prompt or "Untitled multimodal request"
    response = {
        "text": "Text stream mapped into semantic vectors, intent bands, and command candidates.",
        "image": "Image layers scanned for composition, contrast, dominant forms, and annotation targets.",
        "audio": "Audio envelope analyzed for cadence, energy, silence windows, and transcript confidence.",
        "video": "Video sequence indexed for scene shifts, keyframes, visual rhythm, and timeline anchors.",
    }[modality]
    return {
        "id": uuid.uuid4().hex,
        "type": modality,
        "summary": response,
        "input_preview": prompt,
        "confidence": 0.92,
        "signals": [
            {"label": "clarity", "value": 91},
            {"label": "novelty", "value": 78},
            {"label": "actionability", "value": 86},
        ],
        "annotations": [
            {"title": "Context vector", "body": "Primary intent resolved with high semantic agreement."},
            {"title": "Safety pass", "body": "No blocked media patterns found in this local demo pipeline."},
            {"title": "Next action", "body": "Ready for refinement, export, or collaborative review."},
        ],
        "generated_at": int(time.time()),
    }


def process_upload(path: str, modality: str, original_name: str) -> dict[str, Any]:
    file_path = Path(path)
    time.sleep(0.35)
    mime_type = mimetypes.guess_type(original_name)[0] or "application/octet-stream"
    return {
        "filename": original_name,
        "stored_as": file_path.name,
        "type": modality,
        "mime_type": mime_type,
        "bytes": file_path.stat().st_size if file_path.exists() else 0,
        "summary": f"{modality.title()} artifact staged for AETHER analysis.",
    }


def create_app() -> tuple[Flask, Any]:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(
        SECRET_KEY=os.environ.get("AETHER_SECRET_KEY", "aether-dev-secret"),
        MAX_CONTENT_LENGTH=64 * 1024 * 1024,
        UPLOAD_FOLDER=str(UPLOAD_DIR),
        CELERY_BROKER_URL=os.environ.get("CELERY_BROKER_URL", ""),
        CELERY_RESULT_BACKEND=os.environ.get("CELERY_RESULT_BACKEND", ""),
    )

    if CORS is not None:
        CORS(app, resources={r"/api/*": {"origins": os.environ.get("AETHER_CORS_ORIGIN", "*")}})
    else:
        @app.after_request
        def add_cors_headers(response: Response) -> Response:
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
            return response

    if Limiter is not None and get_remote_address is not None:
        Limiter(get_remote_address, app=app, default_limits=["180 per minute", "4000 per day"])

    socketio_enabled = SocketIO is not None and not env_flag("AETHER_DISABLE_SOCKETIO") and not env_flag("VERCEL")
    socketio = (
        SocketIO(app, cors_allowed_origins="*", async_mode="threading", logger=False, engineio_logger=False)
        if socketio_enabled
        else None
    )
    task_queue = LocalTaskQueue()
    celery = make_celery(app)
    celery_process_upload = None
    if celery is not None:
        @celery.task(name="aether.process_upload")
        def celery_process_upload(path: str, modality: str, original_name: str) -> dict[str, Any]:
            return process_upload(path, modality, original_name)

    @app.errorhandler(BadRequest)
    def handle_bad_request(exc: BadRequest) -> tuple[Response, int]:
        return jsonify({"error": exc.description}), 400

    @app.errorhandler(413)
    def handle_file_too_large(_: Exception) -> tuple[Response, int]:
        return jsonify({"error": "File exceeds the 64MB upload limit."}), 413

    @app.get("/")
    def index() -> str:
        theme = request.cookies.get("aether_theme") or session.get("theme") or "dark"
        if theme not in {"dark", "light"}:
            theme = "dark"
        asset_version = os.environ.get("AETHER_ASSET_VERSION") or str(int((BASE_DIR / "static" / "js" / "app.js").stat().st_mtime))
        return render_template(
            "index.html",
            theme=theme,
            socketio_enabled=socketio is not None,
            asset_version=asset_version,
        )

    @app.get("/health")
    def health() -> Response:
        return jsonify(
            {
                "ok": True,
                "name": "AETHER Multimodal Console",
                "socketio": socketio is not None,
                "celery": celery is not None,
            }
        )

    @app.post("/api/multimodal")
    def multimodal_endpoint() -> Response:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise BadRequest("Expected JSON body with type and payload.")
        modality = validate_modality(data.get("type"))
        payload = data.get("payload", "")
        if isinstance(payload, str) and len(payload) > 8000:
            raise BadRequest("Text payload is too large for the live endpoint.")
        result = synthesize_response(modality, payload)
        if socketio is not None:
            socketio.emit("analysis:complete", result, room=data.get("room") or "global")
        return jsonify(result)

    @app.get("/api/stream")
    def stream_endpoint() -> Response:
        prompt = (request.args.get("prompt") or "Generate a multimodal synthesis.").strip()[:500]
        words = (
            "AETHER core online. "
            f"Interpreting {prompt}. "
            "Routing context through text image audio and video channels. "
            "The interface is now ready for collaborative refinement."
        ).split()

        def generate() -> Any:
            for index, word in enumerate(words):
                payload = json.dumps({"token": word, "index": index})
                yield f"event: token\ndata: {payload}\n\n"
                time.sleep(0.06)
            yield "event: done\ndata: {\"ok\": true}\n\n"

        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    @app.post("/api/upload")
    def upload_endpoint() -> Response:
        if "file" not in request.files:
            raise BadRequest("Missing multipart file field named file.")
        uploaded = request.files["file"]
        if not uploaded.filename:
            raise BadRequest("Upload filename is empty.")
        original_name = secure_filename(uploaded.filename)
        modality = request.form.get("type") or infer_modality_from_file(original_name, uploaded.content_type)
        modality = validate_modality(modality)
        validate_upload(original_name, modality)

        stored_name = f"{uuid.uuid4().hex}_{original_name}"
        target = UPLOAD_DIR / stored_name
        uploaded.save(target)

        if celery is not None and celery_process_upload is not None:
            task_id = celery_process_upload.delay(str(target), modality, original_name).id
        else:
            task_id = task_queue.delay(process_upload, str(target), modality, original_name)

        return jsonify(
            {
                "id": uuid.uuid4().hex,
                "task_id": task_id,
                "status": "queued",
                "type": modality,
                "filename": original_name,
                "url": f"/uploads/{stored_name}",
                "bytes": target.stat().st_size,
            }
        )

    @app.get("/api/tasks/<task_id>")
    def task_status(task_id: str) -> Response:
        result = task_queue.get(task_id)
        if result is None:
            return jsonify({"task_id": task_id, "status": "unknown", "payload": {}}), 404
        return jsonify({"task_id": task_id, "status": result.status, "payload": result.payload})

    @app.route("/api/theme", methods=["GET", "POST"])
    def theme_endpoint() -> Response:
        if request.method == "GET":
            theme = request.cookies.get("aether_theme") or session.get("theme") or "dark"
            return jsonify({"theme": theme if theme in {"dark", "light"} else "dark"})
        data = request.get_json(silent=True) or {}
        theme = data.get("theme")
        if theme not in {"dark", "light"}:
            raise BadRequest("theme must be dark or light")
        session["theme"] = theme
        response = make_response(jsonify({"theme": theme}))
        response.set_cookie("aether_theme", theme, max_age=60 * 60 * 24 * 365, samesite="Lax")
        return response

    @app.get("/uploads/<path:filename>")
    def uploaded_file(filename: str) -> Response:
        return send_from_directory(UPLOAD_DIR, filename)

    if socketio is not None:
        @socketio.on("connect")
        def on_connect() -> None:
            join_room("global")
            emit("presence", {"status": "connected", "room": "global"})

        @socketio.on("room:join")
        def on_join(data: dict[str, Any]) -> None:
            room = str(data.get("room") or "global")[:80]
            join_room(room)
            emit("presence", {"status": "joined", "room": room}, room=room, include_self=False)

        @socketio.on("room:leave")
        def on_leave(data: dict[str, Any]) -> None:
            room = str(data.get("room") or "global")[:80]
            leave_room(room)
            emit("presence", {"status": "left", "room": room}, room=room, include_self=False)

        @socketio.on("cursor:move")
        def on_cursor(data: dict[str, Any]) -> None:
            room = str(data.get("room") or "global")[:80]
            emit("cursor:move", data, room=room, include_self=False)

        @socketio.on("modality:change")
        def on_modality(data: dict[str, Any]) -> None:
            room = str(data.get("room") or "global")[:80]
            emit("modality:change", data, room=room, include_self=False)

    return app, socketio


app, socketio = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    if socketio is not None:
        socketio.run(app, host="127.0.0.1", port=port, debug=debug, use_reloader=False, allow_unsafe_werkzeug=True)
    else:
        app.run(host="127.0.0.1", port=port, debug=debug, use_reloader=False)
