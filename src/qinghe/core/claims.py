"""授权只读查单。本模块只有读方法，不提供审批、修改、支付。"""
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from qinghe.utils.audit import AuditLog

EMPLOYEE, FINANCE = "employee", "finance"
# 拒绝话术统一，避免通过不同提示探测别人的单号是否存在；真实原因只写入审计日志
DENY_MESSAGE = "无权查询该单据，或单据不存在。本次请求已记录。"


@dataclass(frozen=True)
class User:
    id: str
    name: str
    role: str


@dataclass(frozen=True)
class Claim:
    id: str
    owner_id: str
    title: str
    amount: float
    submitted: str
    status: str
    missing: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class AccessResult:
    allowed: bool
    message: str = ""
    claim: Claim | None = None


def can_read(user: User, claim: Claim) -> bool:
    """默认拒绝：只有明确列出的情况才放行。"""
    if user.role == FINANCE:
        return True
    return user.role == EMPLOYEE and claim.owner_id == user.id


class ClaimService:
    def __init__(self, data_dir: Path, audit: AuditLog):
        self.audit = audit
        with open(Path(data_dir) / "users.yaml", encoding="utf-8") as f:
            self._users = {u["id"]: User(**u) for u in yaml.safe_load(f)["users"]}
        with open(Path(data_dir) / "claims.yaml", encoding="utf-8") as f:
            self._claims = {
                c["id"]: Claim(
                    id=c["id"],
                    owner_id=c["owner_id"],
                    title=c["title"],
                    amount=c["amount"],
                    submitted=str(c["submitted"]),
                    status=c["status"],
                    missing=tuple(c["missing"]),
                    note=c["note"],
                )
                for c in yaml.safe_load(f)["claims"]
            }

    @property
    def users(self) -> list[User]:
        return list(self._users.values())

    def owner_name(self, claim: Claim) -> str:
        owner = self._users.get(claim.owner_id)
        return owner.name if owner else claim.owner_id

    def list_visible(self, actor_id: str) -> list[Claim]:
        user = self._users.get(actor_id)
        if user is None:
            return []
        return [c for c in self._claims.values() if can_read(user, c)]

    def lookup(self, actor_id: str, claim_id: str) -> AccessResult:
        """按单号查询。先写审计日志再返回；日志写失败会抛异常，不会泄露数据。"""
        start = time.perf_counter()
        claim_id = claim_id.strip().upper()[:64]
        user = self._users.get(actor_id)
        claim = self._claims.get(claim_id)
        if user is None:
            reason = "unknown_actor"
        elif claim is None:
            reason = "claim_not_found"
        elif not can_read(user, claim):
            reason = "forbidden"
        else:
            reason = "ok"
        allowed = reason == "ok"
        self.audit.record(
            action="claim_lookup",
            actor_id=actor_id,
            actor_name=user.name if user else "",
            claim_id=claim_id,
            result="allow" if allowed else "deny",
            reason=reason,
            elapsed_ms=round((time.perf_counter() - start) * 1000, 2),
        )
        if not allowed:
            return AccessResult(False, DENY_MESSAGE)
        return AccessResult(True, claim=claim)
