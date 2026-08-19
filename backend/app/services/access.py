"""访问主体：登录用户（JWT）/ 兼容旧版管理员令牌。"""

from dataclasses import dataclass

from fastapi import Header, HTTPException

from app.config import settings
from app.services.admin_auth import admin_token_configured
from app.services.auth_store import ROLE_ADMIN, UserRecord, decode_access_token


@dataclass
class Actor:
    is_admin: bool = False
    user_id: str = ""
    username: str = ""
    role: str = ""

    @property
    def identity(self) -> str:
        return self.user_id or self.username or ("admin" if self.is_admin else "")


def _is_admin_token(token: str | None) -> bool:
    if not admin_token_configured():
        return False
    provided = (token or "").strip()
    return bool(provided) and provided == settings.admin_token.strip()


def _bearer_token(authorization: str | None) -> str:
    if not authorization:
        return ""
    parts = authorization.strip().split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return ""


def actor_from_user(user: UserRecord) -> Actor:
    return Actor(
        is_admin=(user.role == ROLE_ADMIN),
        user_id=user.id,
        username=user.username,
        role=user.role,
    )


async def get_actor(
    authorization: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> Actor:
    # 1) JWT（正式登录）
    token = _bearer_token(authorization)
    if token:
        user = decode_access_token(token)
        if user:
            return actor_from_user(user)
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")

    # 2) 兼容旧管理员令牌
    if _is_admin_token(x_admin_token):
        return Actor(
            is_admin=True,
            user_id=(x_user_id or "").strip() or "admin",
            username="admin",
            role=ROLE_ADMIN,
        )

    # 3) 兼容旧私人书库钥匙（逐步废弃）
    legacy = (x_user_id or "").strip()
    if legacy:
        return Actor(is_admin=False, user_id=legacy, username="", role="user")

    return Actor()


async def require_login(
    authorization: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> Actor:
    actor = await get_actor(
        authorization=authorization,
        x_admin_token=x_admin_token,
        x_user_id=x_user_id,
    )
    if actor.is_admin or actor.user_id:
        return actor
    raise HTTPException(status_code=401, detail="请先登录")


async def require_uploader_dep(
    authorization: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> Actor:
    actor = await require_login(
        authorization=authorization,
        x_admin_token=x_admin_token,
        x_user_id=x_user_id,
    )
    return actor


async def require_admin_user(
    authorization: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> Actor:
    actor = await get_actor(
        authorization=authorization,
        x_admin_token=x_admin_token,
        x_user_id=None,
    )
    if not actor.is_admin:
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return actor


def assert_can_access_novel(novel, actor: Actor) -> None:
    visibility = getattr(novel, "visibility", "public") or "public"
    owner_id = getattr(novel, "owner_id", "") or ""
    if visibility != "private":
        return
    if actor.is_admin:
        return
    if actor.user_id and actor.user_id == owner_id:
        return
    raise HTTPException(status_code=403, detail="这是私人书库作品，无权访问")


def assert_can_edit_novel(novel, actor: Actor) -> None:
    """编辑人物档案等写操作：私人书仅主人；公共书仅管理员。"""
    assert_can_access_novel(novel, actor)
    if actor.is_admin:
        return
    visibility = getattr(novel, "visibility", "public") or "public"
    owner_id = getattr(novel, "owner_id", "") or ""
    if visibility == "private" and actor.user_id and actor.user_id == owner_id:
        return
    raise HTTPException(
        status_code=403,
        detail="无权编辑该作品的人物档案（公共书仅管理员可改，私人书仅本人可改）",
    )


def assert_can_access_job(job, actor: Actor) -> None:
    if actor.is_admin:
        return
    owner = getattr(job, "owner_id", "") or ""
    if actor.user_id and owner and actor.user_id == owner:
        return
    raise HTTPException(status_code=403, detail="无权访问该准备任务")


def assert_can_access_story(series, actor: Actor) -> None:
    """同人文存档仅作者本人可见；管理员可查看全部。"""
    if actor.is_admin:
        return
    owner = getattr(series, "owner_id", "") or ""
    if actor.user_id and owner and actor.user_id == owner:
        return
    raise HTTPException(status_code=403, detail="这是其他用户的存档，无权访问")
