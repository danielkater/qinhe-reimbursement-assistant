from pathlib import Path

import pytest

from qinghe.core.claims import ClaimService
from qinghe.core.policy import load_active_policy
from qinghe.utils.audit import AuditLog

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
HANDOFF_MESSAGE = "制度中未找到，建议转人工"


@pytest.fixture
def audit(tmp_path):
    return AuditLog(tmp_path / "audit.jsonl")


@pytest.fixture
def service(audit):
    return ClaimService(DATA_DIR, audit)


@pytest.fixture
def policy():
    return load_active_policy(DATA_DIR)
