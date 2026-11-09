"""Part D, tasks 5-8: tools, the ReAct refund agent, safe stopping and crash-resume.

Run from the repository folder (after part_a_store_validate.py):   python src/part_d_agent.py
"""
import glob
import json
import os
import sqlite3
import sys
import time
import uuid
from datetime import datetime

import pandas as pd

sys.path.insert(0, ".")
import mock_llm
from part_d_policy_rag import retrieve

LEDGER_PATH = "outputs/refund_ledger.jsonl"
RUNS_PATH = "outputs/agent_runs.jsonl"
STATE_DIR = "state"

SYSTEM_PROMPT = ("You are a refund AGENT for Instamart. Use the tools get_order, search_policy, "
                 "calculate_refund, issue_refund and escalate_to_human. When you are finished, reply "
                 "with STATUS: explanation, where STATUS is REFUND, ESCALATE, DENIED or NOT_FOUND.")


# ================================================================ task 5: the five tools
def get_order(order_id):
    conn = sqlite3.connect("data/instamart.db")
    row = conn.execute("SELECT order_id, status, order_amount, delivery_minutes "
                       "FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    conn.close()
    if row is None:
        return {"error": "order not found"}
    delivery = None if row[3] is None or pd.isna(row[3]) else float(row[3])
    return {"order_id": row[0], "status": row[1], "order_amount": float(row[2]),
            "delivery_minutes": delivery}


def search_policy(query):
    return {"results": retrieve(query)}


def calculate_refund(order_amount, delivery_minutes):
    """More than 45 minutes -> refund 100% of the order amount, otherwise 0."""
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0.0}


def _write_ledger(tool_name, order_id, refund_inr):
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    line = {"timestamp": datetime.now().isoformat(timespec="seconds"), "tool": tool_name,
            "order_id": order_id, "refund_inr": refund_inr}
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _write_ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _write_ledger("escalate_to_human", order_id, refund_inr)


TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order in the database by its order id.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"]}},
    {"name": "search_policy", "description": "Search the policy documents and return the best matching passages.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                    "required": ["query"]}},
    {"name": "calculate_refund", "description": "Compute the refund amount from the order amount and delivery minutes.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"},
                                                      "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue an automatic refund of up to INR 1000 and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"},
                                                      "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human lead and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"},
                                                      "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]

TOOL_MAP = {"get_order": get_order, "search_policy": search_policy,
            "calculate_refund": calculate_refund, "issue_refund": issue_refund,
            "escalate_to_human": escalate_to_human}


# ================================================================ tasks 6-8: the agent loop
class SimulatedCrash(Exception):
    def __init__(self, run_id):
        super().__init__("simulated crash")
        self.run_id = run_id


def _save_checkpoint(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(os.path.join(STATE_DIR, f"{state['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(state, f)


def _finish(state, status, final_answer):
    amounts = [s["observation"]["refund_inr"] for s in state["steps"]
               if s["tool"] == "calculate_refund" and "refund_inr" in s["observation"]]
    return {"run_id": state["run_id"], "status": status, "final_answer": final_answer,
            "steps": state["steps"], "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"], "refund_inr": amounts[-1] if amounts else 0.0}


def _handoff(state, status):
    last = state["steps"][-1]["tool"] if state["steps"] else "none"
    return (f"Handing over to a human agent: the run stopped ({status}) after {state['llm_calls']} model "
            f"calls and {len(state['steps'])} tool steps; last tool used: {last}.")


def _loop(state, crash_after_tools=None):
    start = time.monotonic()
    while state["llm_calls"] < state["max_iterations"]:
        if time.monotonic() - start >= state["time_limit_seconds"]:           # (c) time limit
            return _finish(state, "FALLBACK_TIMEOUT", _handoff(state, "FALLBACK_TIMEOUT"))

        response = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += response.usage.total_tokens
        if state["tokens_used"] > state["token_limit"]:                       # (b) token limit
            return _finish(state, "FALLBACK_TOKEN_LIMIT", _handoff(state, "FALLBACK_TOKEN_LIMIT"))

        message = response.choices[0].message
        if not message.tool_calls:                                            # final answer
            text = message.content
            return _finish(state, text.split(":")[0].strip(), text)

        call = message.tool_calls[0]
        state["messages"].append({"role": "assistant", "content": None,
                                  "tool_calls": [{"id": call.id, "name": call.name,
                                                  "arguments": call.arguments}]})
        try:
            observation = TOOL_MAP[call.name](**call.arguments)
        except Exception as error:                                            # a failing tool is an observation
            observation = {"error": str(error)}
        state["messages"].append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                  "content": json.dumps(observation)})
        state["steps"].append({"tool": call.name, "arguments": call.arguments, "observation": observation})
        state["tools_executed"].append(call.name)
        state["step"] = len(state["tools_executed"])
        _save_checkpoint(state)                                               # after EVERY observation

        if crash_after_tools and state["step"] == crash_after_tools:
            raise SimulatedCrash(state["run_id"])

    return _finish(state, "FALLBACK_MAX_ITERATIONS", _handoff(state, "FALLBACK_MAX_ITERATIONS"))


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, crash_after_tools=None):
    state = {"run_id": f"{scenario_id}_{uuid.uuid4().hex[:8]}", "scenario_id": scenario_id,
             "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user_message}],
             "tokens_used": 0, "step": 0, "llm_calls": 0, "tools_executed": [], "steps": [],
             "max_iterations": max_iterations, "token_limit": token_limit,
             "time_limit_seconds": time_limit_seconds}
    return _loop(state, crash_after_tools)


def resume_agent(run_id):
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8") as f:
        state = json.load(f)
    return _loop(state)


def log_run(scenario_id, result):
    os.makedirs("outputs", exist_ok=True)
    line = {"scenario_id": scenario_id, "status": result["status"],
            "final_answer": result["final_answer"], "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"], "steps": result["steps"]}
    with open(RUNS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


# ================================================================ demonstrations
if __name__ == "__main__":
    # start from a clean slate so the output files always describe exactly one run
    for path in ([RUNS_PATH, LEDGER_PATH, os.path.join(STATE_DIR, "crash_demo_ledger.jsonl")]
                 + glob.glob(os.path.join(STATE_DIR, "*.json"))):
        if os.path.exists(path):
            os.remove(path)

    scenarios = pd.read_csv("data/raw/agent_scenarios.csv")
    print("=== the seven scenarios ===")
    print(f"{'id':<4}{'status':<26}{'refund INR':>11}{'calls':>7}{'tokens':>8}   expected")
    for s in scenarios.itertuples(index=False):
        mock_llm.reset()
        result = run_agent(s.message, s.scenario_id)
        log_run(s.scenario_id, result)
        expected = "FALLBACK_MAX_ITERATIONS" if s.expected_outcome == "FALLBACK" else s.expected_outcome
        ok = result["status"] == expected and abs(result["refund_inr"] - s.expected_refund_inr) < 0.01
        print(f"{s.scenario_id:<4}{result['status']:<26}{result['refund_inr']:>11.2f}"
              f"{result['llm_calls']:>7}{result['tokens_used']:>8}   "
              f"{expected} {s.expected_refund_inr:.2f}  {'OK' if ok else 'MISMATCH'}")

    s1 = scenarios[scenarios["scenario_id"] == "S1"].iloc[0]

    print("\n=== stop safely: demonstrations on S1 ===")
    mock_llm.reset()
    r = run_agent(s1.message, "S1", token_limit=200)
    print("token_limit=200          ->", r["status"], f"({r['llm_calls']} calls, {r['tokens_used']} tokens)")
    mock_llm.reset()
    r = run_agent(s1.message, "S1", time_limit_seconds=0)
    print("time_limit_seconds=0     ->", r["status"], f"({r['llm_calls']} calls)")
    print("hand-off summary:", r["final_answer"])

    print("\n=== crash and resume on S1 ===")
    LEDGER_PATH = "state/crash_demo_ledger.jsonl"          # keep the demo out of the main ledger
    mock_llm.reset()
    try:
        run_agent(s1.message, "S1", crash_after_tools=2)
    except SimulatedCrash as crash:
        print("simulated crash after the 2nd tool; run id:", crash.run_id)
        result = resume_agent(crash.run_id)
    with open(os.path.join(STATE_DIR, f"{result['run_id']}.json"), encoding="utf-8") as f:
        executed = json.load(f)["tools_executed"]
    print("tools executed across both parts:", executed)
    print("final status:", result["status"], "refund:", f"{result['refund_inr']:.2f}")
    print("tools executed exactly once each:", len(executed) == 4 and len(set(executed)) == 4)
    with open(LEDGER_PATH, encoding="utf-8") as f:
        print("lines in the demo ledger (must be 1):", len(f.readlines()))
