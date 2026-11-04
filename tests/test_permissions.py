import json

import pytest

from qinghe.core.claims import DENY_MESSAGE, ClaimService

XIAOLIN, XIAOWANG, ZHOUMIN = "E1001", "E1002", "F2001"
XIAOLIN_CLAIM, XIAOWANG_CLAIM, OTHER_ORG_CLAIM = "BX-2026-0101", "BX-2026-0103", "BX-2026-0201"


@pytest.mark.parametrize(
    "actor, claim_id, reason",
    [
        (XIAOLIN, XIAOWANG_CLAIM, "forbidden"),
        (XIAOWANG, XIAOLIN_CLAIM, "forbidden"),
        (XIAOLIN, " bx-2026-0103 ", "forbidden"),
        (XIAOLIN, OTHER_ORG_CLAIM, "cross_org"),
        (ZHOUMIN, OTHER_ORG_CLAIM, "cross_org"),  # 财务也不能跨组织
    ],
)
def test_cross_user_and_cross_org_lookup_is_denied_and_logged(service, audit, actor, claim_id, reason):
    result = service.lookup(actor, claim_id)

    assert not result.allowed
    assert result.claim is None
    event = audit.read_all()[-1]
    assert event["actor_id"] == actor
    assert event["result"] == "deny"
    assert event["reason"] == reason


def test_authorized_lookup_shows_status_meaning_within_org(service):
    assert service.lookup(XIAOLIN, XIAOLIN_CLAIM).claim.owner_id == XIAOLIN
    assert service.lookup(ZHOUMIN, XIAOWANG_CLAIM).allowed
    assert {c.owner_id for c in service.list_visible(XIAOLIN)} == {XIAOLIN}
    assert {c.owner_id for c in service.list_visible(ZHOUMIN)} == {XIAOLIN, XIAOWANG}

    # 状态要带含义和更新时间，避免把“审核中”理解成已批准或已付款
    claim = service.lookup(XIAOLIN, "BX-2026-0102").claim
    assert claim.status == "财务审核中"
    assert "不代表已批准或已付款" in claim.status_meaning
    assert claim.updated_at


def test_denials_look_the_same_and_log_stays_minimal(service, audit):
    unknown_actor = service.lookup("HACKER", XIAOLIN_CLAIM)
    missing_claim = service.lookup(XIAOLIN, "BX-0000-0000")
    forbidden = service.lookup(XIAOLIN, XIAOWANG_CLAIM)
    free_text = service.lookup(XIAOLIN, "小林的身份证号")

    assert service.list_visible("HACKER") == []
    for result in (unknown_actor, missing_claim, forbidden, free_text):
        assert not result.allowed
        assert result.claim is None
        assert result.message == DENY_MESSAGE
    # 用户看到相同提示，真实原因只留在审计日志里
    events = audit.read_all()
    assert [e["reason"] for e in events] == [
        "unknown_actor",
        "claim_not_found",
        "forbidden",
        "claim_not_found",
    ]

    # 日志最小化：不存姓名、输入原文和单据内容，白名单外的字段写不进去
    logged = json.dumps(events, ensure_ascii=False)
    for private in ("小林", "HACKER", "身份证", "深圳出差差旅费", "4520"):
        assert private not in logged
    with pytest.raises(ValueError):
        audit.record(action="policy_question", question="我的报销问题")

    # 只读：服务没有审批、修改、支付类方法
    public_methods = {n for n in dir(ClaimService) if not n.startswith("_")}
    assert public_methods == {"users", "owner_name", "list_visible", "lookup"}
