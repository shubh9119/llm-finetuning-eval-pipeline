"""Automated LLM & RAG Evaluation Quality Gate for CI/CD Pipelines."""
from dataclasses import dataclass
from typing import List, Dict, Any

VALID_PATTERN_IDS = {f"P{i:02d}" for i in range(1, 13)}

@dataclass
class EvalMetrics:
    hallucination_rate: float
    faithfulness: float
    context_recall: float
    p95_latency_ms: float

def evaluate_ci_quality_gate(
    metrics: EvalMetrics,
    max_hallucination: float = 0.05,
    min_faithfulness: float = 0.90,
    min_context_recall: float = 0.85,
    max_p95_latency_ms: float = 500.0,
) -> Dict[str, Any]:
    """Block CI/CD model deployment if any LLM/RAG quality metric violates production thresholds."""
    failures: List[str] = []
    if metrics.hallucination_rate > max_hallucination:
        failures.append(f"hallucination_rate {metrics.hallucination_rate:.3f} > {max_hallucination:.3f}")
    if metrics.faithfulness < min_faithfulness:
        failures.append(f"faithfulness {metrics.faithfulness:.3f} < {min_faithfulness:.3f}")
    if metrics.context_recall < min_context_recall:
        failures.append(f"context_recall {metrics.context_recall:.3f} < {min_context_recall:.3f}")
    if metrics.p95_latency_ms > max_p95_latency_ms:
        failures.append(f"p95_latency_ms {metrics.p95_latency_ms:.1f} > {max_p95_latency_ms:.1f}")
    return {
        "passed": len(failures) == 0,
        "BlockedDeployment": len(failures) > 0,
        "violations": failures,
    }

def validate_diagnosis_pattern_id(pattern_id: str) -> bool:
    """Verify that a diagnosed failure pattern ID belongs to the canonical P01-P12 taxonomy."""
    return (pattern_id or "").strip().upper() in VALID_PATTERN_IDS
