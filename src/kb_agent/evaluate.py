"""
读取examples/cases/acceptance.json测试用例输出JSON报告
"""

import argparse
import json
import sys
from pathlib import Path

from kb_agent.agent import run_agent
from kb_agent.knowledge import KnowledgeBase, LocalTools
from kb_agent.model import create_model, load_configuration
from kb_agent.schemas import HistoryMessage


def check_case(case: dict, report: dict) -> list[str]:
    """
    比对运行结果和测试用例
    """
    failures = []
    result, trace = report["result"], report["trace"]
    
    # Status check
    if result["status"] not in case["expected_status"]:
        failures.append(f"Unexpected status: {result['status']}")
        
    # 文档引用检查 expected_docs / forbidden_docs （T07）
    cited = {citation["doc_id"] for citation in result["citations"]}
    if not set(case.get("expected_docs", [])).issubset(cited):
        failures.append("Missing expected source citations.")
    if set(case.get("forbidden_docs", [])) & cited:
        failures.append("Cited a removed document.")
    
    
    # 文档内容检查 expected_text / forbidden_text
    explanation = result["answer"]
    if case.get("expect_conflict"):
        explanation += "\n" + "\n".join(result["conflicts"])
    for text in case.get("expected_text", []):
        if text not in explanation:
            failures.append(f"Answer is missing expected text: {text}")
    for text in case.get("forbidden_text", []):
        if text in result["answer"]:
            failures.append(f"Answer retained stale text: {text}")
            
    # 冲突检查，对应T02      
    if case.get("expect_conflict") and (not result["conflicts"] or not result["follow_up"]):
        failures.append("Conflict and clarification question are required.")   
    if case.get("expect_conflict"):  # 检查是否把所有冲突的source document都读了
        reads = {t["arguments"].get("doc_id") for t in trace if t["action"] == "read_document" and t["observation"].get("ok")}
        if not set(case["expected_docs"]).issubset(reads):
            failures.append("Both conflicting sources must be read to check applicability.")
    
    # 检查模型在finish前是否进行足够次数的搜索
    searches = [t for t in trace if t["action"] == "search_documents" and t["observation"].get("ok")]
    # T03资料外问题，应该至少两次查询
    if len({t["arguments"]["query"] for t in searches}) < case.get("minimum_searches", 0): 
        failures.append("Expected a reformulated search before stopping.")
        
    # 故障恢复检查，对应T08
    if case.get("expect_recovery"):
        failed = [i for i, t in enumerate(trace) if t["observation"].get("error") == "read_unavailable"]
        recovered = any(t["action"] == "read_document" and t["observation"].get("ok") for t in trace[failed[0] + 1:]) if failed else False
        if not recovered:
            failures.append("Expected a model-selected recovery after the injected read failure.")
    
    # 所有用例共用：检查实际调用次数没有超过运行预算
    limits = report["metadata"]["limits"]
    if report["counts"]["model_calls"] > limits["max_model_calls"] or report["counts"]["tool_calls"] > limits["max_tool_calls"]:
        failures.append("Exceeded the execution budget.")
    return failures


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    
    parser = argparse.ArgumentParser(description="Run real-model acceptance cases; no scripted fallback.")
    parser.add_argument("--cases", nargs="+", help="Case IDs; omit to run all cases")
    parser.add_argument("--output", type=Path, default=Path("examples/live"))
    args = parser.parse_args()
    cases = json.loads(Path("examples/cases/acceptance.json").read_text(encoding="utf-8"))
    
    if args.cases:
        unknown = set(args.cases) - {case["id"] for case in cases}
        if unknown:
            parser.error(f"Unknown case IDs: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in args.cases]
    
    config, defaults = load_configuration()
    model = create_model(config)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    
    for case in cases:
        print(f"Running {case['id']}: {case['name']}")
        knowledge = KnowledgeBase(case.get("documents", "data/documents.json"))
        report = run_agent(case["question"], model, LocalTools(knowledge, case.get("failure_mode", "none")),
                           history=[HistoryMessage(**item) for item in case.get("history", [])],
                           limits=defaults, model_name=config["model"], run_kind="live_model")
        report["metadata"]["provider"] = config["provider"]
        failures = check_case(case, report)
        report.update(case_id=case["id"], case_name=case["name"], check_failures=failures)
        (args.output / f"{case['id']}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        
        summaries.append({"id": case["id"], "status": report["result"]["status"], "passed": not failures,
                          "failures": failures, "counts": report["counts"], "trace": f"{case['id']}.json"})
        print(f"{case['id']}: {'PASS' if not failures else 'FAIL'} ({report['result']['status']})")
        
        if report["stop_reason"] == "model_error":
            print("Stopped: model connection failed. Remaining cases were not run.")
            break
    
    summary = {"run_kind": "live_model", "requested": [c["id"] for c in cases], "results": summaries,
               "review_note": "Automated checks are necessary but do not prove semantic entailment. Inspect quotations and claims."}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    
    return 0 if len(summaries) == len(cases) and all(item["passed"] for item in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
