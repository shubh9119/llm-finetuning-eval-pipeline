from quality_gate import EvalMetrics, evaluate_ci_quality_gate, validate_diagnosis_pattern_id
from rag_diagnostics import build_system_prompt, PATTERNS

def test_quality_gate_passes_healthy_model():
    m = EvalMetrics(hallucination_rate=0.032, faithfulness=0.94, context_recall=0.91, p95_latency_ms=380.0)
    res = evaluate_ci_quality_gate(m)
    assert res["passed"] is True
    assert res["BlockedDeployment"] is False
    assert res["violations"] == []

def test_quality_gate_blocks_high_hallucination():
    m = EvalMetrics(hallucination_rate=0.12, faithfulness=0.92, context_recall=0.89, p95_latency_ms=410.0)
    res = evaluate_ci_quality_gate(m)
    assert res["passed"] is False
    assert res["BlockedDeployment"] is True
    assert len(res["violations"]) == 1

def test_diagnostics_patterns_and_prompt():
    assert len(PATTERNS) == 12
    prompt = build_system_prompt()
    assert "P01" in prompt and "P12" in prompt
    assert validate_diagnosis_pattern_id("P01") is True
    assert validate_diagnosis_pattern_id("P99") is False
