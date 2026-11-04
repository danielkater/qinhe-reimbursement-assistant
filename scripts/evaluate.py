"""用样本问题计算制度引用准确率：python scripts/evaluate.py [样本文件]"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

from qinghe.core.policy import load_active_policy
from qinghe.core.qa import PolicyQA
from qinghe.utils.config import load_config


def main() -> int:
    sample = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "examples" / "eval_questions.yaml"
    config = load_config()
    qa = PolicyQA(
        load_active_policy(ROOT / config["paths"]["data_dir"]),
        config["handoff"]["message"],
        top_k=config["retrieval"]["top_k"],
        keep_ratio=config["retrieval"]["keep_ratio"],
    )
    questions = yaml.safe_load(sample.read_text(encoding="utf-8"))["questions"]
    answerable = [q for q in questions if q["expected"]]
    no_basis = [q for q in questions if not q["expected"]]

    cited_right = handoff_right = 0
    for item in questions:
        cited = [c.id for c in qa.ask(item["q"]).citations]
        if item["expected"]:
            ok = bool(cited) and cited[0] == item["expected"]
            cited_right += ok
        else:
            ok = not cited
            handoff_right += ok
        if not ok:
            print(f"未通过：{item['q']}  期望={item['expected'] or '转人工'}  实际={cited or '转人工'}")

    print(f"制度引用准确率：{cited_right}/{len(answerable)}")
    print(f"无依据转人工率：{handoff_right}/{len(no_basis)}")
    return 0 if cited_right == len(answerable) and handoff_right == len(no_basis) else 1


if __name__ == "__main__":
    sys.exit(main())
