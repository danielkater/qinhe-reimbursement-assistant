import pytest

from qinghe.core.claims import DENY_MESSAGE, ClaimService

XIAOLIN, XIAOWANG, ZHOUMIN = "E1001", "E1002", "F2001"
XIAOLIN_CLAIM, XIAOWANG_CLAIM = "BX-2026-0101", "BX-2026-0103"


@pytest.mark.parametrize(
    "actor, claim_id",
    [(XIAOLIN, XIAOWANG_CLAIM), (XIAOWANG, XIAOLIN_CLAIM), (XIAOLIN, " bx-2026-0103 ")],
)
def test_cross_user_lookup_is_denied_and_logged(service, audit, actor, claim_id):
    result = service.lookup(actor, claim_id)

    assert not result.allowed
    assert result.claim is None
    event = audit.read_all()[-1]
    assert event["actor_id"] == actor
    assert event["result"] == "deny"
    assert event["reason"] == "forbidden"


def test_authorized_lookup_and_visibility(service):
    assert service.lookup(XIAOLIN, XIAOLIN_CLAIM).claim.owner_id == XIAOLIN
    assert service.lookup(ZHOUMIN, XIAOWANG_CLAIM).allowed
    assert {c.owner_id for c in service.list_visible(XIAOLIN)} == {XIAOLIN}
    assert {c.owner_id for c in service.list_visible(ZHOUMIN)} == {XIAOLIN, XIAOWANG}


def test_unknown_identity_and_missing_claim_get_same_denial(service, audit):
    unknown_actor = service.lookup("HACKER", XIAOLIN_CLAIM)
    missing_claim = service.lookup(XIAOLIN, "BX-0000-0000")
    forbidden = service.lookup(XIAOLIN, XIAOWANG_CLAIM)

    assert service.list_visible("HACKER") == []
    for result in (unknown_actor, missing_claim, forbidden):
        assert not result.allowed
        assert result.claim is None
        assert result.message == DENY_MESSAGE
    # 用户看到相同提示，真实原因只留在审计日志里
    assert [e["reason"] for e in audit.read_all()] == [
        "unknown_actor",
        "claim_not_found",
        "forbidden",
    ]
    # 只读：服务没有审批、修改、支付类方法
    public_methods = {n for n in dir(ClaimService) if not n.startswith("_")}
    assert public_methods == {"users", "owner_name", "list_visible", "lookup"}
