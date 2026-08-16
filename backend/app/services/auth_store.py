"""用户账号：SQLite 存储，预置账号、无开放注册。"""

from __future__ import annotations

import secrets
import sqlite3
import string
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
from passlib.context import CryptContext

from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ROLE_ADMIN = "admin"
ROLE_USER = "user"


@dataclass
class UserRecord:
    id: str
    username: str
    role: str
    display_name: str = ""
    is_active: bool = True


def _db_path() -> Path:
    path = Path(settings.auth_db_path)
    if not path.is_absolute():
        # 相对 backend 工作目录
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                display_name TEXT NOT NULL DEFAULT '',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return pwd_context.verify(password, password_hash)
    except Exception:
        return False


def generate_password(length: int = 10) -> str:
    alphabet = string.ascii_letters + string.digits
    # 保证至少各有一个大小写和数字，便于分发又不太弱
    while True:
        pwd = "".join(secrets.choice(alphabet) for _ in range(length))
        if (
            any(c.islower() for c in pwd)
            and any(c.isupper() for c in pwd)
            and any(c.isdigit() for c in pwd)
        ):
            return pwd


def get_user_by_username(username: str) -> UserRecord | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, username, role, display_name, is_active FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if not row:
        return None
    return UserRecord(
        id=row["id"],
        username=row["username"],
        role=row["role"],
        display_name=row["display_name"] or "",
        is_active=bool(row["is_active"]),
    )


def get_user_by_id(user_id: str) -> UserRecord | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, username, role, display_name, is_active FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    if not row:
        return None
    return UserRecord(
        id=row["id"],
        username=row["username"],
        role=row["role"],
        display_name=row["display_name"] or "",
        is_active=bool(row["is_active"]),
    )


def get_password_hash(username: str) -> str | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    return row["password_hash"] if row else None


def count_users() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()
    return int(row["c"] if row else 0)


def upsert_user(
    *,
    user_id: str,
    username: str,
    password: str,
    role: str,
    display_name: str = "",
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    pw_hash = hash_password(password)
    with _connect() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE users
                SET password_hash = ?, role = ?, display_name = ?, is_active = 1
                WHERE username = ?
                """,
                (pw_hash, role, display_name, username),
            )
        else:
            conn.execute(
                """
                INSERT INTO users (id, username, password_hash, role, display_name, is_active, created_at)
                VALUES (?, ?, ?, ?, ?, 1, ?)
                """,
                (user_id, username, pw_hash, role, display_name, now),
            )
        conn.commit()


def authenticate(username: str, password: str) -> UserRecord | None:
    user = get_user_by_username(username)
    if not user or not user.is_active:
        return None
    pw_hash = get_password_hash(username)
    if not pw_hash or not verify_password(password, pw_hash):
        return None
    return user


def create_access_token(user: UserRecord) -> str:
    secret = (settings.jwt_secret or "").strip() or "change-me-aitongrenwen-jwt-secret"
    expire = datetime.now(timezone.utc) + timedelta(hours=max(1, settings.jwt_expire_hours))
    payload = {
        "sub": user.id,
        "username": user.username,
        "role": user.role,
        "exp": expire,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_access_token(token: str) -> UserRecord | None:
    secret = (settings.jwt_secret or "").strip() or "change-me-aitongrenwen-jwt-secret"
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except Exception:
        return None
    user_id = str(payload.get("sub") or "")
    if not user_id:
        return None
    user = get_user_by_id(user_id)
    if not user or not user.is_active:
        return None
    return user


def seed_preset_accounts(force_reset_passwords: bool = False) -> list[dict]:
    """
    预置 admin + user01..user20。
    若库中已有对应用户且 force_reset_passwords=False，则跳过已存在账号（不改密）。
    返回本次生成/更新的明文账号清单（仅应写入本地 credentials 文件）。
    """
    init_db()
    credentials: list[dict] = []

    admin_user = settings.admin_username.strip() or "admin"
    admin_pwd = (settings.admin_password or "").strip() or generate_password(12)

    with _connect() as conn:
        exists_admin = conn.execute(
            "SELECT id FROM users WHERE username = ?", (admin_user,)
        ).fetchone()

    if force_reset_passwords or not exists_admin:
        upsert_user(
            user_id="admin",
            username=admin_user,
            password=admin_pwd,
            role=ROLE_ADMIN,
            display_name="管理员",
        )
        credentials.append(
            {
                "username": admin_user,
                "password": admin_pwd,
                "role": ROLE_ADMIN,
                "display_name": "管理员",
            }
        )

    for i in range(1, 21):
        username = f"user{i:02d}"
        with _connect() as conn:
            exists = conn.execute(
                "SELECT id FROM users WHERE username = ?", (username,)
            ).fetchone()
        if exists and not force_reset_passwords:
            continue
        password = generate_password(10)
        upsert_user(
            user_id=username,
            username=username,
            password=password,
            role=ROLE_USER,
            display_name=f"用户{i:02d}",
        )
        credentials.append(
            {
                "username": username,
                "password": password,
                "role": ROLE_USER,
                "display_name": f"用户{i:02d}",
            }
        )

    return credentials
