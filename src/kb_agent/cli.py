import argparse
import json
import sys
from pathlib import Path

from pydantic import TypeAdapter

from kb_agent.agent import run_agent, unfinished
from kb_agent.knowledge import KnowledgeBase, LocalTools
from kb_agent.model import create_model, load_configuration
from kb_agent.schemas import HistoryMessage, Limits


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="企业知识问答 Agent"
    )
    parser.add_argument(
        "--question",
        required=True,
        help="你想查询的制度问题",
    )
    parser.add_argument("--documents", default="data/documents.json")
    parser.add_argument("--history", type=Path, help="JSON list of user/assistant messages")
    parser.add_argument("--trace", type=Path, help="Write the complete execution record as JSON")
    parser.add_argument("--max-model-calls", type=int)
    parser.add_argument("--max-tool-calls", type=int)
    parser.add_argument("--failure-mode", choices=["none", "read_once"], default="none")
    args = parser.parse_args()

    try:
        config, defaults = load_configuration()
        limits = Limits(
            max_model_calls=args.max_model_calls if args.max_model_calls is not None else defaults.max_model_calls,
            max_tool_calls=args.max_tool_calls if args.max_tool_calls is not None else defaults.max_tool_calls,
        )
        history = TypeAdapter(list[HistoryMessage]).validate_json(args.history.read_bytes()) if args.history else []
        report = run_agent(
            args.question, create_model(config), LocalTools(KnowledgeBase(args.documents), args.failure_mode),
            history=history, limits=limits, model_name=config["model"], run_kind="live_model",
        )
        report["metadata"]["provider"] = config["provider"]
    except (ValueError, OSError) as exc:
        # Never print configuration objects, API keys, or provider exception bodies.
        report = {"question": args.question, **unfinished("configuration_error", str(exc)), "trace": [],
                  "counts": {"model_calls": 0, "tool_calls": 0}}
    
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.trace.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    
    print(json.dumps({k: v for k, v in report.items() if k != "trace"}, ensure_ascii=False, indent=2))
    return 2 if report["result"]["status"] in {"incomplete", "tool_error"} else 0
