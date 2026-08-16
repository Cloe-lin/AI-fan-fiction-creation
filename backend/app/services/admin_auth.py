"""管理员鉴权：导入小说等写操作。"""

from fastapi import Header, HTTPException

from app.config import settings

_PLACEHOLDER_ADMIN = {"", "your_admin_token", "change_me", "admin"}


def admin_token_configured() -> bool:
    token = (settings.admin_token or "").strip()
    return token not in _PLACEHOLDER_ADMIN


async def require_admin(x_admin_token: str | None = Header(default=None, alias="X-Admin-Token")):
    if not admin_token_configured():
        raise HTTPException(
            status_code=503,
            detail="未配置管理员令牌（请在 backend/.env 设置 ADMIN_TOKEN）",
        )
    provided = (x_admin_token or "").strip()
    if not provided or provided != settings.admin_token.strip():
        raise HTTPException(status_code=401, detail="需要管理员权限")
    return True
