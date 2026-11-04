"""授权只读查单。本模块只有读方法，不提供审批、修改、支付。"""
import re
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from qinghe.utils.audit import AuditLog

EMPLOYEE, FINANCE = "employee", "finance"
# 拒绝话术统一，避免通过不同提示探测别人的单号是否存在；真实原因只写入审计日志
DENY_MESSAGE = "无权查询该单据，或单据不存在。本次请求已记录。"
CLAIM_ID_FORMAT = re.compile(r"[A-Z]+-\d{4}-\d{4}")
# 状态含义是演示用的草案，正式口径待财务核对。未列出的状态不做猜测
STATUS_MEANINGS = {
    "部门审批中": "部门负责人正在审批，不代表已批准或已付款。",
    "财务审核中": "财务正在审核，不代表已批准或已付款。",
    "待补件": "材料不全，等待员工补齐，不代表已批准或已付款。",
    "已打款": "财务已完成打款。",
}
UNKNOWN_STATUS_MEANING = "该状态的含义待财务确认。"


@dataclass(frozen=True)
class User:
    id: str
    name: str
    role: str
    org: str


@dataclass(frozen=True)
class Claim:
    id: str
    owner_id: str
    org: str
    title: str
    amount: float
    submitted: str
    updated_at: str
    status: str
    missing: tuple[str, ...]
    note: str

    @property
    def status_meaning(self) -> str:
        return STATUS_MEANINGS.get(self.status, UNKNOWN_STATUS_MEANING)


@dataclass(frozen=True)
class AccessResult:
    allowed: bool
    message: str = ""
    claim: Claim | None = None


def can_read(user: User, claim: Claim) -> bool:
    """默认拒绝：先看组织，再按角色放行明确列出的情况。"""
    if user.org != claim.org:
        return False
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
                    org=c["org"],
                    title=c["title"],
                    amount=c["amount"],
                    submitted=str(c["submitted"]),
                    updated_at=str(c["updated_at"]),
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
        elif user.org != claim.org:
            reason = "cross_org"
        elif not can_read(user, claim):
            reason = "forbidden"
        else:
            reason = "ok"
        allowed = reason == "ok"
        # 输入框内容不可信，不符合单号格式或身份未知时不把原文写入日志
        self.audit.record(
            action="claim_lookup",
            actor_id=user.id if user else "unknown",
            claim_id=claim_id if CLAIM_ID_FORMAT.fullmatch(claim_id) else "invalid",
            result="allow" if allowed else "deny",
            reason=reason,
            elapsed_ms=round((time.perf_counter() - start) * 1000, 2),
        )
        if not allowed:
            return AccessResult(False, DENY_MESSAGE)
        return AccessResult(True, claim=claim)
