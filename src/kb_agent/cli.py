import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="企业知识问答 Agent"
    )
    parser.add_argument(
        "--question",
        required=True,
        help="你想查询的制度问题",
    )
    args = parser.parse_args()

    # 尚未接入检索或模型。
    result = {
        "status": "incomplete",
        "question": args.question,
        "answer": None,
        "citations": [],
        "gaps": ["尚未实现知识库检索和模型决策"],
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))