from __future__ import annotations

import asyncio
import logging
import queue
import secrets
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Cookie, FastAPI, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from product_app.app.auth import hash_password, new_token, verify_password
from product_app.app.classifiers import IntentClassifier
from product_app.app.config import settings
from product_app.app.db import SQLiteStore
from product_app.app.llm import LLMConfig, SoulHarborLLM
from product_app.app.memory.session_context import messages_for_classifier
from product_app.app.memory.service import MemoryService


logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_DIR.parent / "templates"
STATIC_DIR = APP_DIR.parent / "static"
SPA_DIR = STATIC_DIR / "spa"
SPA_INDEX = SPA_DIR / "index.html"


app = FastAPI(title="SoulHarbor Product App", version="0.2.0")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _spa_enabled() -> bool:
    return SPA_INDEX.is_file()


def _serve_spa() -> Response:
    return FileResponse(SPA_INDEX)


_DB = SQLiteStore(settings.db_path)
_MEMORY = MemoryService(_DB)
_POST_TURN_EXECUTOR = ThreadPoolExecutor(
    max_workers=settings.post_turn_workers,
    thread_name_prefix="soulharbor_post_turn",
)
# LLM calls are serialized by the GPU RLock regardless of thread pool size.
# A small pool is honest about the serial nature while allowing a few
# concurrent non-GPU operations (DB queries, token counting).
_LLM_EXECUTOR = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="soulharbor_llm",
)
_ADMIN_TOKENS: set[str] = set()


def _require_user(auth: Optional[str]) -> Dict[str, Any]:
    if not auth:
        raise PermissionError("no_auth")
    user = _DB.get_user_by_token(auth)
    if not user:
        raise PermissionError("bad_auth")
    return user


def _require_admin(admin: Optional[str]) -> None:
    if not admin or admin not in _ADMIN_TOKENS:
        raise PermissionError("admin")

def _format_for_classifier(messages: List[Dict[str, str]], *, max_user_turns: int = 3) -> str:
    return messages_for_classifier(messages, max_user_turns=max_user_turns)


def _schedule_post_turn(
    *,
    user_id: int,
    conversation_id: int,
    sid: str,
    user_message_id: int,
    assistant_message_id: int,
    user_position: int,
    assistant_position: int,
    user_created_at: int,
    assistant_created_at: int,
    user_text: str,
    assistant_text: str,
    is_consult: int,
) -> None:
    def _run_sync() -> None:
        _MEMORY.run_post_turn(
            user_id=user_id,
            conversation_id=conversation_id,
            sid=sid,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
            user_position=user_position,
            assistant_position=assistant_position,
            user_created_at=user_created_at,
            assistant_created_at=assistant_created_at,
            user_text=user_text,
            assistant_text=assistant_text,
            is_consult=is_consult,
        )

    _POST_TURN_EXECUTOR.submit(_run_sync)


def _finalize_chat_turn(
    *,
    conversation_id: int,
    sid: str,
    user_id: int,
    user_message_id: int,
    user_position: int,
    user_created_at: int,
    user_text: str,
    assistant_text: str,
    is_consult: int,
    assistant_message_id: Optional[int] = None,
    assistant_position: Optional[int] = None,
    assistant_created_at: Optional[int] = None,
) -> None:
    assistant = (assistant_text or "").strip() or "我在。你愿意和我说说发生了什么吗？"
    if assistant_message_id is not None:
        _DB.update_message_content(assistant_message_id, assistant)
        if assistant_position is None or assistant_created_at is None:
            rec = _DB.get_message_record(assistant_message_id)
            if rec is None:
                return
            assistant_position = rec.position
            assistant_created_at = rec.created_at
            assistant_message_id = rec.message_id
    else:
        row = _DB.append_message(conversation_id, "assistant", assistant)
        assistant_message_id = row.message_id
        assistant_position = row.position
        assistant_created_at = row.created_at

    route = "consult" if is_consult == 1 else "chat"
    _DB.append_turn_metrics(
        conversation_id,
        is_consult=is_consult,
        route=route,
    )
    _schedule_post_turn(
        user_id=user_id,
        conversation_id=conversation_id,
        sid=sid,
        user_message_id=user_message_id,
        assistant_message_id=int(assistant_message_id),
        user_position=user_position,
        assistant_position=int(assistant_position),
        user_created_at=user_created_at,
        assistant_created_at=int(assistant_created_at),
        user_text=user_text,
        assistant_text=assistant,
        is_consult=is_consult,
    )


def _chat_stream_producer(
    *,
    conversation_id: int,
    sid: str,
    user_id: int,
    user_message_id: int,
    user_position: int,
    user_created_at: int,
    user_text: str,
    chunk_queue: "queue.Queue[Optional[str]]",
    assistant_message_id: int,
    assistant_position: int,
    assistant_created_at: int,
    is_consult: Optional[int] = None,
    model_messages: Optional[List[Dict[str, str]]] = None,
    prepare_fn: Optional[Any] = None,
) -> None:
    """Run LLM streaming in a background thread; persist even if the client disconnects."""
    chunks: List[str] = []
    consult = 1 if is_consult is None else int(is_consult)
    try:
        if prepare_fn is not None:
            consult, model_messages = prepare_fn()
        if not model_messages:
            raise RuntimeError("empty_model_messages")
        for chunk in _LLM.generate_stream(model_messages, is_consult=consult):  # type: ignore[union-attr]
            s = str(chunk)
            if not s:
                continue
            chunks.append(s)
            chunk_queue.put(s)
    except Exception:
        logger.exception("chat stream generation failed")
        if not chunks:
            # Send error to client only; do NOT save error text to DB
            # (would pollute conversation history for future turns).
            chunk_queue.put("\n（生成失败，请稍后再试）")
    finally:
        assistant_text = "".join(chunks).strip()
        _finalize_chat_turn(
            conversation_id=conversation_id,
            sid=sid,
            user_id=user_id,
            user_message_id=user_message_id,
            user_position=user_position,
            user_created_at=user_created_at,
            user_text=user_text,
            assistant_text=assistant_text,
            is_consult=consult,
            assistant_message_id=assistant_message_id,
            assistant_position=assistant_position,
            assistant_created_at=assistant_created_at,
        )
        chunk_queue.put(None)


def _stream_chat_response(
    *,
    conversation_id: int,
    sid: str,
    user_id: int,
    user_message_id: int,
    user_position: int,
    user_created_at: int,
    user_text: str,
    is_consult: Optional[int] = None,
    model_messages: Optional[List[Dict[str, str]]] = None,
    prepare_fn: Optional[Any] = None,
) -> StreamingResponse:
    assistant_row = _DB.append_message(conversation_id, "assistant", "")
    assistant_message_id = assistant_row.message_id
    chunk_queue: queue.Queue[Optional[str]] = queue.Queue()
    threading.Thread(
        target=_chat_stream_producer,
        kwargs={
            "conversation_id": conversation_id,
            "sid": sid,
            "user_id": user_id,
            "user_message_id": user_message_id,
            "user_position": user_position,
            "user_created_at": user_created_at,
            "user_text": user_text,
            "is_consult": is_consult,
            "model_messages": model_messages,
            "prepare_fn": prepare_fn,
            "chunk_queue": chunk_queue,
            "assistant_message_id": assistant_message_id,
            "assistant_position": assistant_row.position,
            "assistant_created_at": assistant_row.created_at,
        },
        name=f"soulharbor_chat_stream_{conversation_id}",
        daemon=False,
    ).start()

    def _iter_stream() -> Any:
        try:
            while True:
                item = chunk_queue.get()
                if item is None:
                    break
                yield item
        except GeneratorExit:
            # Client navigated away; producer thread keeps running and will persist the reply.
            pass

    return StreamingResponse(
        _iter_stream(),
        media_type="text/plain; charset=utf-8",
        headers={
            # Critical behind AutoDL / nginx: disable response buffering so
            # tokens reach the browser as they are generated.
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.on_event("startup")
def _startup() -> None:
    global _INTENT, _LLM
    _DB.init()
    _INTENT = IntentClassifier(
        run_dir=settings.classifier_run,
        encoder_base=settings.encoder_base,
        max_length=settings.max_length,
        device=settings.classifier_device,
    )
    _LLM = SoulHarborLLM(
        LLMConfig(
            base=settings.llm_base,
            adapter=settings.llm_adapter,
            adapter_scale=settings.llm_adapter_scale,
            casual_adapter_scale=settings.llm_casual_adapter_scale,
            system=settings.llm_system,
            load_4bit=settings.llm_load_4bit,
            device=settings.llm_device,
            max_new_tokens=settings.llm_max_new_tokens,
        )
    )
    _MEMORY.set_llm(_LLM)

@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "llm_base": settings.llm_base,
            "llm_adapter": settings.llm_adapter,
            "classifier_run": settings.classifier_run,
            "db_path": settings.db_path,
        }
    )


# ---------- Pages ----------
@app.get("/", response_class=HTMLResponse)
async def root() -> Response:
    return RedirectResponse("/app")


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> Response:
    if _spa_enabled():
        return _serve_spa()
    return templates.TemplateResponse(request, "login.html", {})


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request) -> Response:
    if _spa_enabled():
        return _serve_spa()
    return templates.TemplateResponse(request, "register.html", {})


@app.get("/app", response_class=HTMLResponse)
async def app_page(request: Request, auth: Optional[str] = Cookie(default=None)) -> Response:
    try:
        user = _require_user(auth)
    except PermissionError:
        return RedirectResponse("/login")
    if _spa_enabled():
        return _serve_spa()
    return templates.TemplateResponse(
        request,
        "app.html",
        {"username": user["username"]},
    )


@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request) -> Response:
    if _spa_enabled():
        return _serve_spa()
    return templates.TemplateResponse(request, "admin_login.html", {})


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, admin: Optional[str] = Cookie(default=None)) -> Response:
    try:
        _require_admin(admin)
    except PermissionError:
        return RedirectResponse("/admin/login")
    return templates.TemplateResponse(request, "admin.html", {})


# ---------- Auth APIs ----------
@app.get("/api/me")
async def me_api(auth: Optional[str] = Cookie(default=None)) -> JSONResponse:
    try:
        user = _require_user(auth)
    except PermissionError:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    return JSONResponse({"ok": True, "username": user["username"], "user_id": int(user["id"])})


@app.post("/api/register")
async def register_api(request: Request) -> JSONResponse:
    payload = await request.json()
    username = str(payload.get("username") or "").strip()
    password = str(payload.get("password") or "")
    if not username or len(username) < 3:
        return JSONResponse({"ok": False, "error": "bad_username"}, status_code=400)
    if len(password) < 6:
        return JSONResponse({"ok": False, "error": "bad_password"}, status_code=400)
    if _DB.get_user_by_username(username):
        return JSONResponse({"ok": False, "error": "username_taken"}, status_code=400)

    uid = _DB.create_user(username, hash_password(password))
    token = new_token()
    _DB.create_session(token, uid)

    sid = str(uuid.uuid4())
    _DB.get_or_create_conversation(sid, user_id=int(uid))

    resp = JSONResponse({"ok": True})
    resp.set_cookie("auth", token, httponly=True, samesite="lax")
    resp.set_cookie("sid", sid, httponly=True, samesite="lax")
    return resp


@app.post("/api/login")
async def login_api(request: Request) -> JSONResponse:
    payload = await request.json()
    username = str(payload.get("username") or "").strip()
    password = str(payload.get("password") or "")
    u = _DB.get_user_by_username(username)
    if not u or not verify_password(password, u["password_hash"]):
        return JSONResponse({"ok": False, "error": "invalid_credentials"}, status_code=401)

    token = new_token()
    _DB.create_session(token, int(u["id"]))

    # Rotate sid on login to avoid cross-user sid cookie causing 403.
    sid = str(uuid.uuid4())
    _DB.get_or_create_conversation(sid, user_id=int(u["id"]))
    resp = JSONResponse({"ok": True})
    resp.set_cookie("auth", token, httponly=True, samesite="lax")
    resp.set_cookie("sid", sid, httponly=True, samesite="lax")
    return resp


@app.post("/api/logout")
async def logout_api(auth: Optional[str] = Cookie(default=None)) -> JSONResponse:
    if auth:
        _DB.delete_session(auth)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("auth")
    resp.delete_cookie("sid")
    return resp


# ---------- Chat APIs (user-facing; no debug UI) ----------
@app.get("/api/conversations")
async def api_conversations(auth: Optional[str] = Cookie(default=None)) -> JSONResponse:
    try:
        user = _require_user(auth)
    except PermissionError:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)

    if user["id"] == 0:
        items = _DB.list_conversations(limit=100)
    else:
        items = _DB.list_conversations_for_user(int(user["id"]), limit=100)
    return JSONResponse({"ok": True, "conversations": items})


@app.post("/api/conversation/new")
async def api_conversation_new(
    auth: Optional[str] = Cookie(default=None),
    sid: Optional[str] = Cookie(default=None),
) -> JSONResponse:
    try:
        user = _require_user(auth)
    except PermissionError:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)

    uid = None if user["id"] == 0 else int(user["id"])
    current = (sid or "").strip()

    # GPT-style: if already on an empty draft, stay put instead of spawning another.
    if current:
        conv = _DB.get_conversation(current)
        if conv and (user["id"] == 0 or conv.get("user_id") in (None, uid)):
            if _DB.count_messages(int(conv["id"])) == 0:
                resp = JSONResponse({"ok": True, "sid": current, "reused": True})
                resp.set_cookie("sid", current, httponly=True, samesite="lax")
                return resp

    if uid is not None:
        _DB.delete_empty_conversations_for_user(uid)

    new_sid = str(uuid.uuid4())
    _DB.get_or_create_conversation(new_sid, user_id=uid)
    resp = JSONResponse({"ok": True, "sid": new_sid, "reused": False})
    resp.set_cookie("sid", new_sid, httponly=True, samesite="lax")
    return resp

@app.post("/api/conversation/select")
async def api_conversation_select(request: Request, auth: Optional[str] = Cookie(default=None), sid: Optional[str] = Cookie(default=None)) -> JSONResponse:
    try:
        user = _require_user(auth)
    except PermissionError:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    payload = await request.json()
    target = str(payload.get("sid") or "").strip()
    if not target:
        return JSONResponse({"ok": False, "error": "missing_sid"}, status_code=400)

    conv = _DB.get_conversation(target)
    if conv and user["id"] != 0 and conv.get("user_id") not in (None, int(user["id"])):
        return JSONResponse({"ok": False, "error": "forbidden"}, status_code=403)

    uid = None if user["id"] == 0 else int(user["id"])
    # Drop abandoned empty drafts when leaving them for a real conversation.
    if uid is not None:
        prev = (sid or "").strip()
        if prev and prev != target:
            prev_conv = _DB.get_conversation(prev)
            if prev_conv and prev_conv.get("user_id") in (None, uid) and _DB.count_messages(int(prev_conv["id"])) == 0:
                _DB.delete_conversation(prev)
        _DB.delete_empty_conversations_for_user(uid, keep_sid=target)

    _DB.get_or_create_conversation(target, user_id=uid)
    resp = JSONResponse({"ok": True, "sid": target})
    resp.set_cookie("sid", target, httponly=True, samesite="lax")
    return resp


@app.post("/api/conversation/rename")
async def api_conversation_rename(request: Request, auth: Optional[str] = Cookie(default=None)) -> JSONResponse:
    try:
        user = _require_user(auth)
    except PermissionError:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    payload = await request.json()
    sid = str(payload.get("sid") or "").strip()
    title = str(payload.get("title") or "").strip()
    if not sid:
        return JSONResponse({"ok": False, "error": "missing_sid"}, status_code=400)
    conv = _DB.get_conversation(sid)
    if conv and user["id"] != 0 and conv.get("user_id") not in (None, int(user["id"])):
        return JSONResponse({"ok": False, "error": "forbidden"}, status_code=403)
    _DB.set_title(sid, title)
    return JSONResponse({"ok": True})


@app.post("/api/conversation/delete")
async def api_conversation_delete(request: Request, auth: Optional[str] = Cookie(default=None), sid: Optional[str] = Cookie(default=None)) -> JSONResponse:
    try:
        user = _require_user(auth)
    except PermissionError:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    payload = await request.json()
    target_sid = str(payload.get("sid") or "").strip()
    if not target_sid:
        return JSONResponse({"ok": False, "error": "missing_sid"}, status_code=400)
    conv = _DB.get_conversation(target_sid)
    if conv and user["id"] != 0 and conv.get("user_id") not in (None, int(user["id"])):
        return JSONResponse({"ok": False, "error": "forbidden"}, status_code=403)

    _DB.delete_conversation(target_sid)
    if sid and sid == target_sid:
        new_sid = str(uuid.uuid4())
        _DB.get_or_create_conversation(new_sid, user_id=(None if user["id"] == 0 else int(user["id"])))
        resp = JSONResponse({"ok": True, "sid": new_sid, "rotated": True})
        resp.set_cookie("sid", new_sid, httponly=True, samesite="lax")
        return resp
    return JSONResponse({"ok": True, "rotated": False})


@app.get("/api/history")
async def api_history(auth: Optional[str] = Cookie(default=None), sid: Optional[str] = Cookie(default=None)) -> JSONResponse:
    try:
        user = _require_user(auth)
    except PermissionError:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    if not sid:
        return JSONResponse({"ok": True, "messages": []})

    conv = _DB.get_conversation(sid)
    if conv and user["id"] != 0 and conv.get("user_id") not in (None, int(user["id"])):
        return JSONResponse({"ok": False, "error": "forbidden"}, status_code=403)

    cid = _DB.get_or_create_conversation(sid, user_id=(None if user["id"] == 0 else int(user["id"])))
    msgs = _DB.list_messages(cid, limit=200)
    return JSONResponse({"ok": True, "messages": [{"role": m.role, "content": m.content, "created_at": m.created_at} for m in msgs]})


@app.get("/api/memory/settings")
async def api_memory_settings(auth: Optional[str] = Cookie(default=None)) -> JSONResponse:
    try:
        user = _require_user(auth)
    except PermissionError:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    uid = int(user["id"])
    enabled = _MEMORY.is_long_term_active(uid) if uid > 0 else False
    count = _MEMORY.count_active_memories(uid) if uid > 0 else 0
    return JSONResponse({"ok": True, "memory_enabled": enabled, "memory_count": count})


@app.post("/api/memory/enable")
async def api_memory_enable(auth: Optional[str] = Cookie(default=None)) -> JSONResponse:
    try:
        user = _require_user(auth)
    except PermissionError:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    if user["id"] <= 0:
        return JSONResponse({"ok": False, "error": "invalid_user"}, status_code=400)
    _DB.set_memory_enabled(int(user["id"]), True)
    return JSONResponse({"ok": True, "memory_enabled": True})


@app.post("/api/memory/disable")
async def api_memory_disable(auth: Optional[str] = Cookie(default=None)) -> JSONResponse:
    try:
        user = _require_user(auth)
    except PermissionError:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    if user["id"] <= 0:
        return JSONResponse({"ok": False, "error": "invalid_user"}, status_code=400)
    _DB.set_memory_enabled(int(user["id"]), False)
    return JSONResponse({"ok": True, "memory_enabled": False})


@app.post("/api/memory/forget-all")
async def api_memory_forget_all(auth: Optional[str] = Cookie(default=None)) -> JSONResponse:
    try:
        user = _require_user(auth)
    except PermissionError:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    if user["id"] <= 0:
        return JSONResponse({"ok": False, "error": "invalid_user"}, status_code=400)
    n = _MEMORY.forget_all_memories(int(user["id"]))
    _DB.append_memory_event(int(user["id"]), "forget_all", detail={"deleted": n})
    return JSONResponse({"ok": True, "deleted": n})


def _classify_turn(classifier_msgs: List[Dict[str, str]]) -> int:
    cls_text = _format_for_classifier(classifier_msgs, max_user_turns=3)
    intent = _INTENT.predict(cls_text)  # type: ignore[union-attr]
    return int(intent.is_consult)


def _prepare_chat_turn(
    *,
    conversation_id: int,
    sid: str,
    user_id: int,
    user_text: str,
) -> tuple[int, List[Dict[str, str]]]:
    """Classify intent, then build model prompt once (single memory recall)."""
    is_new_session = _DB.count_messages(conversation_id) == 1
    classifier_msgs = _MEMORY.build_classifier_messages(conversation_id)
    is_consult = _classify_turn(classifier_msgs)
    _, model_messages = _MEMORY.build_chat_context(
        conversation_id=conversation_id,
        sid=sid,
        user_id=user_id,
        user_query=user_text,
        is_consult=is_consult,
        is_new_session=is_new_session,
    )
    return is_consult, model_messages


@app.post("/api/chat")
async def api_chat(request: Request, auth: Optional[str] = Cookie(default=None), sid: Optional[str] = Cookie(default=None)) -> JSONResponse:
    try:
        user = _require_user(auth)
    except PermissionError:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)

    payload = await request.json()
    user_text = (payload.get("text") or "").strip()
    if not user_text:
        return JSONResponse({"ok": False, "error": "empty"}, status_code=400)

    if not sid:
        sid = str(uuid.uuid4())

    conv = _DB.get_conversation(sid)
    if conv and user["id"] != 0 and conv.get("user_id") not in (None, int(user["id"])):
        # Do not leak other user's conversation. Rotate silently.
        logger.warning("sid rotated: conversation user_id mismatch for user %s", user["id"])
        sid = str(uuid.uuid4())
        conv = None

    cid = _DB.get_or_create_conversation(sid, user_id=(None if user["id"] == 0 else int(user["id"])))
    user_row = _DB.append_message(cid, "user", user_text)

    title = _DB.get_title(sid)
    if not title:
        auto = user_text.replace("\n", " ").strip()
        auto = auto[:20] + ("…" if len(auto) > 20 else "")
        _DB.set_title(sid, auto)

    is_consult, model_messages = _prepare_chat_turn(
        conversation_id=cid,
        sid=sid,
        user_id=int(user["id"]),
        user_text=user_text,
    )

    assistant = await asyncio.get_event_loop().run_in_executor(
        _LLM_EXECUTOR,
        partial(_LLM.generate, model_messages, is_consult=is_consult),  # type: ignore[union-attr]
    )
    assistant = (assistant or "").strip() or "我在。你愿意和我说说发生了什么吗？"
    assistant_row = _DB.append_message(cid, "assistant", assistant)

    route = "consult" if is_consult == 1 else "chat"
    _DB.append_turn_metrics(
        cid,
        is_consult=is_consult,
        route=route,
    )
    _schedule_post_turn(
        user_id=int(user["id"]),
        conversation_id=cid,
        sid=sid,
        user_message_id=user_row.message_id,
        assistant_message_id=assistant_row.message_id,
        user_position=user_row.position,
        assistant_position=assistant_row.position,
        user_created_at=user_row.created_at,
        assistant_created_at=assistant_row.created_at,
        user_text=user_text,
        assistant_text=assistant,
        is_consult=is_consult,
    )

    resp = JSONResponse({"ok": True, "assistant": assistant})
    resp.set_cookie("sid", sid, httponly=True, samesite="lax")
    return resp


@app.post("/api/chat_stream")
async def api_chat_stream(request: Request, auth: Optional[str] = Cookie(default=None), sid: Optional[str] = Cookie(default=None)) -> Response:
    """
    Streaming variant of /api/chat for a better UX.
    Returns plain text chunks; client assembles them.
    """
    try:
        user = _require_user(auth)
    except PermissionError:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)

    payload = await request.json()
    user_text = (payload.get("text") or "").strip()
    if not user_text:
        return JSONResponse({"ok": False, "error": "empty"}, status_code=400)

    if not sid:
        sid = str(uuid.uuid4())

    conv = _DB.get_conversation(sid)
    if conv and user["id"] != 0 and conv.get("user_id") not in (None, int(user["id"])):
        # Do not leak other user's conversation. Rotate silently.
        logger.warning("sid rotated: conversation user_id mismatch for user %s", user["id"])
        sid = str(uuid.uuid4())
        conv = None

    cid = _DB.get_or_create_conversation(sid, user_id=(None if user["id"] == 0 else int(user["id"])))
    user_row = _DB.append_message(cid, "user", user_text)

    title = _DB.get_title(sid)
    if not title:
        auto = user_text.replace("\n", " ").strip()
        auto = auto[:20] + ("…" if len(auto) > 20 else "")
        _DB.set_title(sid, auto)

    # Return the streaming response immediately. Intent + memory retrieval run
    # inside the producer thread so proxies/browsers get headers/TTFB early;
    # otherwise AutoDL buffering makes the whole reply appear at once.
    uid = int(user["id"])

    def _prepare() -> tuple[int, List[Dict[str, str]]]:
        return _prepare_chat_turn(
            conversation_id=cid,
            sid=sid,
            user_id=uid,
            user_text=user_text,
        )

    resp = _stream_chat_response(
        conversation_id=cid,
        sid=sid,
        user_id=uid,
        user_message_id=user_row.message_id,
        user_position=user_row.position,
        user_created_at=user_row.created_at,
        user_text=user_text,
        prepare_fn=_prepare,
    )
    resp.set_cookie("sid", sid, httponly=True, samesite="lax")
    return resp


# ---------- Admin APIs (anonymous) ----------
@app.post("/admin/api/login")
async def admin_login_api(request: Request) -> JSONResponse:
    payload = await request.json()
    password = str(payload.get("password") or "")
    if password != settings.admin_password:
        return JSONResponse({"ok": False, "error": "invalid"}, status_code=401)
    admin_token = secrets.token_hex(16)
    _ADMIN_TOKENS.add(admin_token)
    resp = JSONResponse({"ok": True})
    resp.set_cookie("admin", admin_token, httponly=True, samesite="lax")
    return resp


@app.post("/admin/api/logout")
async def admin_logout_api(admin: Optional[str] = Cookie(default=None)) -> JSONResponse:
    _ADMIN_TOKENS.discard(admin)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("admin")
    return resp


@app.get("/admin/api/stats")
async def admin_stats(
    admin: Optional[str] = Cookie(default=None),
    date_from: int = 0,
    date_to: int = 0,
) -> JSONResponse:
    try:
        _require_admin(admin)
    except PermissionError:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    return JSONResponse({"ok": True, "today": _DB.stats_today(date_from=date_from, date_to=date_to), "db_path": settings.db_path})


@app.get("/admin/api/triage")
async def admin_triage(
    admin: Optional[str] = Cookie(default=None),
    status: str = "all",
    limit: int = 80,
    date_from: int = 0,
    date_to: int = 0,
) -> JSONResponse:
    try:
        _require_admin(admin)
    except PermissionError:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    items = _DB.list_triage_sessions(
        limit=max(10, min(int(limit), 200)),
        status=status,
        date_from=date_from,
        date_to=date_to,
    )
    return JSONResponse({"ok": True, "items": items})


@app.get("/admin/api/triage_records")
async def admin_triage_records(
    admin: Optional[str] = Cookie(default=None),
    status: str = "all",
    limit: int = 100,
    date_from: int = 0,
    date_to: int = 0,
) -> JSONResponse:
    try:
        _require_admin(admin)
    except PermissionError:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    items = _DB.list_triage_records(
        limit=max(10, min(int(limit), 200)),
        status=status,
        date_from=date_from,
        date_to=date_to,
    )
    return JSONResponse({"ok": True, "items": items})


@app.get("/admin/api/conversation-workspace")
async def admin_conversation_workspace(sid: str, admin: Optional[str] = Cookie(default=None)) -> JSONResponse:
    try:
        _require_admin(admin)
    except PermissionError:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    detail = _DB.get_admin_conversation_workspace(sid)
    if detail is None:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
    return JSONResponse({"ok": True, "detail": detail})


@app.get("/admin/api/conversations")
async def admin_conversations(
    admin: Optional[str] = Cookie(default=None),
    limit: int = 100,
    offset: int = 0,
    date_from: int = 0,
    date_to: int = 0,
) -> JSONResponse:
    try:
        _require_admin(admin)
    except PermissionError:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    items = _DB.list_conversations_with_metrics(
        limit=limit, offset=offset, date_from=date_from, date_to=date_to
    )
    return JSONResponse({"ok": True, "conversations": items})


@app.get("/admin/api/conversations/{sid}")
async def admin_conversation_detail(sid: str, admin: Optional[str] = Cookie(default=None)) -> JSONResponse:
    try:
        _require_admin(admin)
    except PermissionError:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    detail = _DB.get_conversation_detail(sid)
    if detail is None:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
    return JSONResponse({"ok": True, "conversation": detail})


@app.get("/admin/api/users")
async def admin_users(
    admin: Optional[str] = Cookie(default=None),
    limit: int = 50,
    offset: int = 0,
) -> JSONResponse:
    try:
        _require_admin(admin)
    except PermissionError:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    items = _DB.list_users_with_stats(limit=limit, offset=offset)
    return JSONResponse({"ok": True, "users": items})


@app.get("/admin/api/user-profile-workspace")
async def admin_user_profile_workspace(user_id_hash: str, admin: Optional[str] = Cookie(default=None)) -> JSONResponse:
    try:
        _require_admin(admin)
    except PermissionError:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    all_users = _DB.list_users_with_stats(limit=1000, offset=0)
    uid = None
    for u in all_users:
        if u["user_id_hash"] == user_id_hash:
            uid = u["user_id"]
            break
    if uid is None:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
    detail = _MEMORY.get_user_profile(int(uid))
    if detail is None:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
    return JSONResponse({"ok": True, "detail": detail})


@app.get("/admin/api/users/{user_id_hash}/profile")
async def admin_user_profile(user_id_hash: str, admin: Optional[str] = Cookie(default=None)) -> JSONResponse:
    try:
        _require_admin(admin)
    except PermissionError:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    # Resolve hash back to user_id
    all_users = _DB.list_users_with_stats(limit=1000, offset=0)
    uid = None
    for u in all_users:
        if u["user_id_hash"] == user_id_hash:
            uid = u["user_id"]
            break
    if uid is None:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
    profile = _MEMORY.get_user_profile(uid)
    if profile is None:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
    return JSONResponse({"ok": True, "profile": profile})


@app.post("/admin/api/triage/update")
async def admin_triage_update(request: Request, admin: Optional[str] = Cookie(default=None)) -> JSONResponse:
    try:
        _require_admin(admin)
    except PermissionError:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    payload = await request.json()
    sid = str(payload.get("sid") or "").strip()
    if not sid:
        return JSONResponse({"ok": False, "error": "bad_sid"}, status_code=400)
    detail = _DB.get_admin_conversation_workspace(sid)
    if detail is None:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
    conversation_id = int(detail["conversation"]["id"])
    status = str(payload.get("status") or "pending").strip().lower()
    assignee = str(payload.get("assignee") or "").strip()
    note = str(payload.get("note") or "").strip()
    _DB.upsert_triage_record(conversation_id, status=status, assignee=assignee, note=note)
    updated = _DB.get_admin_conversation_workspace(sid)
    return JSONResponse({"ok": True, "detail": updated})
