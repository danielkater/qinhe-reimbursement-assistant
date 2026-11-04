"""Streamlit 入口：streamlit run src/qinghe/main.py"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from qinghe.core.claims import FINANCE, ClaimService
from qinghe.core.llm import build_llm
from qinghe.core.policy import load_active_policy
from qinghe.core.qa import HANDOFF, PolicyQA
from qinghe.utils.audit import AuditLog
from qinghe.utils.config import ROOT, load_config, load_llm_settings

ROLE_LABELS = {"employee": "员工", "finance": "财务"}
RESULT_LABELS = {"allow": "放行", "deny": "拒绝", "answered": "已回答", "handoff": "转人工"}
REASON_LABELS = {
    "ok": "授权通过",
    "forbidden": "越权查询他人单据",
    "cross_org": "跨组织查询",
    "claim_not_found": "单据不存在",
    "unknown_actor": "身份无法识别",
}
EXAMPLES = ["上海住酒店一晚最多报多少？", "发票丢了怎么办？", "健身卡可以报销吗？"]


@st.cache_resource
def load_app():
    config = load_config()
    data_dir = ROOT / config["paths"]["data_dir"]
    audit = AuditLog(ROOT / config["paths"]["audit_log"])
    settings = load_llm_settings()
    llm = build_llm(settings, config["llm"]["timeout_seconds"], config["llm"]["temperature"])
    qa = PolicyQA(
        load_active_policy(data_dir),
        config["handoff"]["message"],
        llm,
        config["retrieval"]["top_k"],
        config["retrieval"]["keep_ratio"],
    )
    return config, audit, qa, ClaimService(data_dir, audit), settings


def fill_question(text: str) -> None:
    st.session_state.question = text


def render_policy_tab(config, audit, qa, actor, use_llm) -> None:
    policy = qa.policy
    st.caption(f"当前有效版本：《{policy.name}》{policy.version}，{policy.effective_date} 起生效")

    columns = st.columns(len(EXAMPLES))
    for column, example in zip(columns, EXAMPLES):
        column.button(example, on_click=fill_question, args=(example,), width="stretch")

    with st.form("qa_form"):
        question = st.text_input("你的问题", key="question", placeholder="例如：报销时限是多久？")
        submitted = st.form_submit_button("提问", type="primary")

    if submitted and question.strip():
        answer = qa.ask(question, use_llm=use_llm)
        audit.record(
            action="policy_question",
            actor_id=actor.id,
            result=answer.status,
            mode=answer.mode,
            citations=[c.id for c in answer.citations],
        )
        st.session_state.answer = answer
        st.session_state.claim_hint = bool(re.search(r"BX-\d+", question, re.I))

    answer = st.session_state.get("answer")
    if answer is None:
        return
    if answer.status == HANDOFF:
        st.warning(answer.text)
        st.caption(config["handoff"]["contact"])
        if st.session_state.get("claim_hint"):
            st.info("查询单据请使用「授权查单」页，系统会先校验你的身份。")
        return
    st.success(f"依据条款：{'、'.join(c.id for c in answer.citations)}")
    st.markdown(answer.text)
    if answer.note:
        st.info(answer.note)
    if answer.mode == "llm":
        with st.expander("查看引用的条款原文"):
            for clause in answer.citations:
                st.markdown(f"**{clause.id} {clause.title}**：{clause.text}")


def render_claim_detail(service, claim) -> None:
    st.markdown(f"#### {claim.id}　{claim.title}")
    left, middle, right = st.columns(3)
    left.metric("状态", claim.status)
    middle.metric("金额（元）", f"{claim.amount:,.2f}")
    right.metric("提交日期", claim.submitted)
    st.info(f"状态含义：{claim.status_meaning}")
    st.write(f"申请人：{service.owner_name(claim)}　｜　状态更新时间：{claim.updated_at}")
    st.caption("数据来源：虚构演示数据。状态的权威来源和正式含义待财务确认。")
    if claim.missing:
        st.warning(f"缺件：{'、'.join(claim.missing)}")
    if claim.note:
        st.write(claim.note)


def render_claims_tab(service, actor) -> None:
    st.caption("只读查询，不提供审批、修改和支付。状态只表示当前阶段，除“已打款”外都不代表已付款。")
    visible = service.list_visible(actor.id)
    title = "本组织单据" if actor.role == FINANCE else "我的单据"
    st.markdown(f"**{title}（{len(visible)} 笔）**")
    st.dataframe(
        [
            {
                "单号": c.id,
                "申请人": service.owner_name(c),
                "事项": c.title,
                "金额（元）": c.amount,
                "状态": c.status,
                "更新时间": c.updated_at,
                "缺件": "、".join(c.missing) or "无",
            }
            for c in visible
        ],
        hide_index=True,
        width="stretch",
    )

    with st.form("lookup_form"):
        claim_id = st.text_input("按单号查询", placeholder="例如：BX-2026-0101")
        submitted = st.form_submit_button("查询", type="primary")
    if submitted and claim_id.strip():
        st.session_state.lookup = (actor.id, service.lookup(actor.id, claim_id))

    stored = st.session_state.get("lookup")
    # 切换身份后不再显示上一个身份的查询结果
    if stored is None or stored[0] != actor.id:
        return
    result = stored[1]
    if result.allowed:
        render_claim_detail(service, result.claim)
    else:
        st.error(result.message)


def render_audit_tab(audit, service) -> None:
    names = {u.id: u.name for u in service.users}
    summary = audit.summary()
    columns = st.columns(5)
    columns[0].metric("查单次数", summary["lookups"])
    columns[1].metric("权限拒绝", summary["denied"])
    avg_ms = summary["avg_lookup_ms"]
    columns[2].metric("平均查单耗时", "<1 ms" if avg_ms < 1 else f"{avg_ms} ms")
    columns[3].metric("制度提问", summary["questions"])
    columns[4].metric("转人工", summary["handoffs"])
    st.caption(
        "以上为本机演示日志的统计，查单耗时只是系统响应时间，不代表线上效果。"
        "日志只存身份编号、单号、结果和原因，不存姓名、提问原文和单据内容。"
    )

    rows = [
        {
            "时间": e["time"],
            "身份": names.get(e.get("actor_id"), e.get("actor_id")),
            "操作": "查单" if e["action"] == "claim_lookup" else "制度提问",
            "对象": e.get("claim_id", ""),
            "结果": RESULT_LABELS.get(e["result"], e["result"]),
            "说明": REASON_LABELS.get(e.get("reason"), "、".join(e.get("citations", []))),
        }
        for e in reversed(audit.read_all())
    ]
    st.dataframe(rows, hide_index=True, width="stretch")


def main() -> None:
    st.set_page_config(page_title="青禾报销助手", page_icon="🌾", layout="wide")
    config, audit, qa, service, settings = load_app()
    st.title(config["app"]["title"])

    with st.sidebar:
        st.header("模拟身份")
        users = {f"{u.name}（{ROLE_LABELS[u.role]}）": u for u in service.users}
        actor = users[st.selectbox("当前身份", list(users))]
        st.caption("演示用的模拟登录，真实身份系统不在本次范围内。")
        st.divider()
        st.header("问答模式")
        use_llm = False
        if qa.llm is not None:
            use_llm = st.toggle(f"使用大模型（{settings.model}）", value=True)
        st.write("大模型模式" if use_llm else "本地关键词检索模式")
        st.divider()
        st.caption("系统边界：不自动审批，不支付，查单数据不发送给大模型。")

    names = ["制度问答", "授权查单"] + (["审计日志"] if actor.role == FINANCE else [])
    tabs = st.tabs(names)
    with tabs[0]:
        render_policy_tab(config, audit, qa, actor, use_llm)
    with tabs[1]:
        render_claims_tab(service, actor)
    if actor.role == FINANCE:
        with tabs[2]:
            render_audit_tab(audit, service)


if __name__ == "__main__":
    main()
