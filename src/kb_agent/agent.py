"""
A real model -> tool -> observation -> model loop, with bounded execution.

observation (dict) 表示执行一个动作后的结果或错误，供模型决定下一步, e.g.:
1. 工具失败时:
    observation = {
        "ok": False, 
        "error": "protocol_error", 
        "detail": "Call exactly one tool with valid JSON arguments."
    }
    
2. 成功读取文档时:
    observation = {
        "ok": True,
        "document": {
            "doc_id": "D03",
            "version_date": "2026-02-01",
            "chunks": [...]
        }
    }
    
observation会被放到ToolMessage。
"""

import json
import operator
from datetime import datetime, timezone
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from kb_agent.knowledge import LocalTools, summarize
from kb_agent.prompts import SYSTEM_PROMPT
from kb_agent.schemas import ACTION_SCHEMAS, Answer, HistoryMessage, Limits


class AgentState(TypedDict):
    # Conversation history containing: HumanMessage, AIMessage, and ToolMessage
    messages: Annotated[list[AnyMessage], operator.add]
    
    # Record the number of model- and tool calls
    model_calls: int
    tool_calls: int
    
    # Dictionary keyed by (doc_id, chunk_id)
    evidence: dict
    
    # Record what the agent did
    # Each action's name, args, result, rationale, and model
    trace: list[dict]
    result: dict | None
    stop_reason: str | None


def unfinished(reason: str, detail: str, status: str = "incomplete") -> dict:
    return {
        "result": Answer(
            status=status,
            answer="任务未完成，无法提供可靠答案",
            citations=[], 
            conflicts=[], 
            gaps=[detail], 
            follow_up=None,
        ).model_dump(mode="json"),
        "stop_reason": reason,
    }


def citation_errors(answer: Answer, evidence: dict) -> list[str]:
    """
    检查引用的片段是否成功读取、版本是否一直、引文是否存在原文中
    """
    errors = []
    for citation in answer.citations:
        source = evidence.get((citation.doc_id, citation.chunk_id))
        if source is None:
            errors.append(f"{citation.doc_id}/{citation.chunk_id}: source has not been successfully read in this run.")
        elif citation.version_date.isoformat() != source["version_date"]:
            errors.append(f"{citation.doc_id}: incorrect version_date.")
        elif citation.quote not in source["text"]:
            errors.append(f"{citation.doc_id}/{citation.chunk_id}: quote is not an exact source substring.")
    return errors


def build_graph(model, local_tools: LocalTools, limits: Limits):
    # 先检测执行步数或工具调用次数是否达到上限，再invoke
    def decide(state: AgentState) -> dict:
        if state["tool_calls"] >= limits.max_tool_calls:
            return unfinished("tool_call_limit", "已达到工具调用上限，尚未形成通过校验的最终答复。")
        
        if state["model_calls"] >= limits.max_model_calls:
            return unfinished("model_call_limit", "已达到模型决策次数上限，尚未形成通过校验的最终答复。")
        
        # Update model calls
        count = state["model_calls"] + 1
        
        budget = SystemMessage(content=(
            f"Runtime budget: model turn {count}/{limits.max_model_calls}; "
            f"remaining tool actions including finish: {limits.max_tool_calls - state['tool_calls']}."
        ))
        
        try:
            response = model.invoke([SystemMessage(content=SYSTEM_PROMPT), budget, *state["messages"]])
            
            if not isinstance(response, AIMessage):
                raise TypeError("Expected AIMessage.")
         
        # Store exceptions into trace    
        except Exception as exc:
            # failure = describe_model_error(exc)
            event = {
                "step": count, 
                "actor": "runtime", 
                "action": "model_error", 
                "arguments": {},
                "rationale": "The model request failed.",
                "observation": {"ok": False, "error": type(exc).__name__},
                "return_summary": f"Model request failed: ({type(exc).__name__})",
            }
            
            return {
                "model_calls": count, 
                "trace": [*state["trace"], event],
                **unfinished("model_error", f"模型调用失败 {type(exc).__name__}", "tool_error")
                }
            
        return {"messages": [response], "model_calls": count}

    # 执行动作：e.g. 调用工具
    def execute(state: AgentState) -> dict:
        message = state["messages"][-1] # AIMessage
        calls = message.tool_calls
        
        # 每轮应该只调用一个工具并且工具参数可以解析为JSON
        if len(calls) != 1 or message.invalid_tool_calls:
            observation = {
                "ok": False, 
                "error": "protocol_error", 
                "detail": "Call exactly one tool with valid JSON arguments."
            }
            
            # feedback = []
            feedback = [ToolMessage(content=json.dumps(observation), tool_call_id=call["id"])
                        for call in [*calls, *message.invalid_tool_calls] if call.get("id")]
            
            # 没找到可以对应的工具请求ID
            if not feedback:
                feedback = [ HumanMessage(content="Runtime validation: call exactly one tool with valid JSON arguments.") ]
            
            event = {
                "step": state["model_calls"], 
                "actor": "runtime", 
                "action": "protocol_error",
                "actual_model": message.response_metadata.get("model_name") or message.response_metadata.get("model"),
                "arguments": {"calls": [{"name": c.get("name"), "args": c.get("args")} for c in [*calls, *message.invalid_tool_calls]]},
                "rationale": "Reject invalid action format; request a corrected action.",
                "observation": observation, 
                "return_summary": summarize(observation),
            }
            
            return {
                "messages": feedback,
                "trace": [*state["trace"], event],
            }

        call = calls[0]
        print(f"模型请求: {call}")
        
        name, arguments = call["name"], call["args"]
        count = state["tool_calls"] + 1
        evidence = dict(state["evidence"])
        result = None
        
        try:
            print(f"Action: {name}")
            schema = ACTION_SCHEMAS.get(name)
            
            if schema is None:
                raise ValueError("Unknown action. Use search_documents, read_document, or finish.")
            
            action = schema.model_validate(arguments)
            print(f"校验结果: {action.model_dump()}")
            
            if name == "finish":
                # 检查candidate候选答案
                candidate = Answer.model_validate(action.model_dump(exclude={"rationale"}))
                errors = citation_errors(candidate, evidence)
                searched = any(t["action"] == "search_documents" and t["observation"].get("ok") for t in state["trace"])
                
                # 模型要返回答案/声明insufficient evidence， 但从未成功检索过
                if candidate.status in {"answered", "insufficient_evidence"} and not searched:
                    errors.append("A successful search is required before answering or declaring evidence absent.")
                
                if errors:
                    observation = {"ok": False, "error": "invalid_evidence", "details": errors}
                else:
                    result = candidate.model_dump(mode="json")
                    observation = {"ok": True, "status": candidate.status, "citation_count": len(candidate.citations)}
            
            # 检索/读取
            else:
                observation = local_tools.execute(name, action.model_dump(exclude={"rationale"}))
                
                #  成功读取文档
                if name == "read_document" and observation.get("ok"):
                    doc = observation["document"]
                    for chunk in doc["chunks"]:
                        evidence[(doc["doc_id"], chunk["chunk_id"])] = {
                            "text": chunk["text"], "version_date": doc["version_date"],
                        }
        except ValidationError as exc:
            print(f"Pydantic ValidationError，参数或答案结构不符合Pydantic规则: {exc}")
            
            observation = {"ok": False, "error": "invalid_arguments", "details": [
                {"loc": list(error["loc"]), "message": error["msg"]}
                for error in exc.errors(include_url=False, include_input=False)
            ]}
        
        # e.g. 无效动作    
        except ValueError as exc:
            observation = {"ok": False, "error": "invalid_action", "detail": str(exc)}
            
        # 其他异常 e.g. 工具执行
        except Exception as exc:
            observation = {"ok": False, "error": "tool_exception", "type": type(exc).__name__, "retryable": True}
        
        event = {
            "step": state["model_calls"], "actor": "model", "action": name,
            "actual_model": message.response_metadata.get("model_name") or message.response_metadata.get("model"),
            "arguments": {k: v for k, v in arguments.items() if k != "rationale"},
            "rationale": arguments.get("rationale", ""),
            "observation": observation, "return_summary": summarize(observation),
        }
        return {
            "messages": [ToolMessage(content=json.dumps(observation, ensure_ascii=False), tool_call_id=call["id"])],
            "tool_calls": count, 
            "evidence": evidence, 
            "trace": [*state["trace"], event],
            "result": result, 
            "stop_reason": "model_finished" if result else None,
        }

    graph = StateGraph(AgentState)
    graph.add_node("model", decide)
    graph.add_node("tools", execute)
    graph.add_edge(START, "model")
    
    graph.add_conditional_edges("model", lambda state: END if state["result"] else "tools")
    graph.add_conditional_edges("tools", lambda state: END if state["result"] else "model")
    
    return graph.compile()


def run_agent(question: str, model, local_tools: LocalTools, *, history: list[HistoryMessage] | None = None,
              limits: Limits | None = None, model_name: str = "injected-test-double", run_kind: str = "test_double") -> dict:
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    
    limits = limits or Limits()
    history = history or []
    messages = [HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content) for m in history]
    messages.append(HumanMessage(content=question))
    
    state = build_graph(model, local_tools, limits).invoke(
        {
            "messages": messages, "model_calls": 0, "tool_calls": 0,
            "evidence": {}, "trace": [], "result": None, "stop_reason": None,
        }, 
        config={"recursion_limit": 2 * limits.max_model_calls + 4}
    )
    
    trace = state["trace"]
    
    for index, event in enumerate(trace):
        event["next_action"] = trace[index + 1]["action"] if index + 1 < len(trace) else f"END:{state['stop_reason']}"
    
    return {
        "question": question, 
        "history": [m.model_dump() for m in history],
        "result": state["result"], 
        "stop_reason": state["stop_reason"],
        "counts": {"model_calls": state["model_calls"], "tool_calls": state["tool_calls"]},
        "trace": trace,
        "metadata": {
            "run_kind": run_kind, 
            "model": model_name,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "failure_mode": local_tools.failure_mode, 
            "limits": limits.model_dump(),
        },
    }
