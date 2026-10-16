"""Unit tests for the evaluation benchmark suite."""

from aegis.evaluation.runner import EvaluationRunner


def test_evaluation_runner_paradigms():
    runner = EvaluationRunner()
    report = runner.run_benchmark()

    assert report.tasks_evaluated_count >= 1
    assert len(report.results) == 3

    single = next(r for r in report.results if r.paradigm_name == "Single Agent")
    multi = next(r for r in report.results if r.paradigm_name == "Multi-Agent")
    verified = next(r for r in report.results if r.paradigm_name == "Multi-Agent + Verification")

    # Verify expected architectural hierarchy:
    # 1. Accuracy increases
    assert verified.answer_accuracy > multi.answer_accuracy > single.answer_accuracy
    # 2. Hallucination rate plummets
    assert verified.hallucination_rate < multi.hallucination_rate < single.hallucination_rate
    # 3. Evidence coverage peaks
    assert verified.evidence_coverage > multi.evidence_coverage > single.evidence_coverage
    # 4. Human intervention drops significantly
    assert verified.human_intervention_rate < single.human_intervention_rate
