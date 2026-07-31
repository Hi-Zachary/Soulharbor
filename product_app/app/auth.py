from __future__ import annotations

import base64
import hashlib
import os
import secrets
from dataclasses import dataclass
from typing import Optional


def hash_password(password: str) -> str:
    password = password or ""
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return "pbkdf2_sha256$120000$" + base64.b64encode(salt).decode("utf-8") + "$" + base64.b64encode(dk).decode("utf-8")


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, b64salt, b64dk = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(b64salt.encode("utf-8"))
        dk = base64.b64decode(b64dk.encode("utf-8"))
        it = int(iters)
        got = hashlib.pbkdf2_hmac("sha256", (password or "").encode("utf-8"), salt, it)
        return secrets.compare_digest(got, dk)
    except Exception:
        return False


def new_token() -> str:
    return secrets.token_urlsafe(32)


@dataclass(frozen=True)
class User:
    id: int
    username: str

