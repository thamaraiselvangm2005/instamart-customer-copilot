"""Part C: support-ticket triage with guardrails, retries, validation, a token budget and routing.

Run from the repository folder:   python src/part_c_triage.py
(needs outputs/customer_segments.csv from Part B)
"""
import json
import os
import random
import re
import sys
import time

import pandas as pd

sys.path.insert(0, ".")                      # so that "import mock_llm" finds the file in the repo root
import mock_llm

CATEGORIES = ["billing", "delivery", "technical", "fraud"]
MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05

# ---------------------------------------------------------------- redaction (order matters)
PATTERNS = [("email", r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]"),
            ("phone", r"\b[6-9]\d{9}\b", "[PHONE]"),
            ("order id", r"ORD-\d{6}", "[ORDER_ID]")]


def redact(text, counts=None):
    for name, pattern, placeholder in PATTERNS:
        text, n = re.subn(pattern, placeholder, text)
        if counts is not None:
            counts[name] += n
    return text


# ---------------------------------------------------------------- the two prompts
ZERO_SHOT = ('CLASSIFY the support ticket as one of: billing, delivery, technical, fraud. '
             'Reply with JSON only: {"category": "...", "reason": "..."}')

FEW_SHOT = ('CLASSIFY the support ticket as one of: billing, delivery, technical, fraud. '
            'Reply with JSON only: {"category": "...", "reason": "..."}\n'
            'EXAMPLES:\n'
            'Ticket: I was charged twice for one order -> {"category": "billing", "reason": "double charge"}\n'
            'Ticket: Someone hacked my account and made unauthorized purchases -> '
            '{"category": "fraud", "reason": "unauthorized access"}\n'
            'Ticket: The app crashes when I open it -> {"category": "technical", "reason": "app crash"}')


# ---------------------------------------------------------------- validation of one answer
def validate(content):
    """Return (status, category). Never guess."""
    try:
        data = json.loads(content)
        if isinstance(data, dict) and data.get("category") in CATEGORIES:
            return "classified_json", data["category"]
        return "flagged_invalid_output", ""             # valid JSON but wrong category (or wrong shape)
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", str(content).lower())
        found = {c for c in CATEGORIES if c in words}
        if len(found) == 1:                              # exactly one allowed category word
            return "classified_keyword", found.pop()
        return "flagged_invalid_output", ""


# ---------------------------------------------------------------- one model call with retries
def call_with_retries(messages, ticket_id, retry_log):
    for retry in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if retry == MAX_RETRIES:
                return None                              # gave up: permanent error
            wait = BASE_DELAY * 2 ** retry * random.uniform(0.9, 1.15)
            retry_log.append((ticket_id, retry, wait))
            time.sleep(wait)


# ---------------------------------------------------------------- the pipeline
def run_pipeline(tickets, system_prompt, token_budget=None, show_retry_log=False):
    mock_llm.reset()                                     # fresh counters for every full run
    random.seed(0)
    retry_log, sent_user_messages, rows = [], [], []
    total_tokens = 0

    for t in tickets.itertuples(index=False):
        text = "" if pd.isna(t.text) else str(t.text)
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id,
               "status": "", "category": "", "tokens_used": 0}

        if text.strip() == "":                                           # guardrail (a)
            row["status"] = "skipped_empty"
        elif mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:               # guardrail (b)
            row["status"] = "flagged_long"
        elif token_budget is not None and total_tokens >= token_budget:  # budget
            row["status"] = "budget_exhausted"
        else:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": redact(text)}]
            sent_user_messages.append(messages[1]["content"])
            response = call_with_retries(messages, t.ticket_id, retry_log)
            if response is None:
                row["status"] = "flagged_invalid_output"                 # never happens with this mock
            else:
                used = response.usage.total_tokens
                total_tokens += used
                row["tokens_used"] = used
                row["status"], row["category"] = validate(response.choices[0].message.content)
        rows.append(row)

    if show_retry_log:
        for ticket_id, retry, wait in retry_log:
            print(f"  retry log: ticket {ticket_id}  retry #{retry}  wait {wait:.4f} s")
    return pd.DataFrame(rows), retry_log, sent_user_messages, total_tokens


def accuracy(result, truth):
    ok = result[result["status"].isin(["classified_json", "classified_keyword"])]
    merged = ok.merge(truth, on="ticket_id")
    return len(ok), int((merged["category"] == merged["true_category"]).sum())


def route(result, segments):
    """Routing rules from the brief."""
    seg = segments.set_index("customer_id")
    out = []
    for r in result.itertuples(index=False):
        name = seg["segment_name"].get(r.customer_id, "") if r.customer_id in seg.index else ""
        premium = bool(seg["is_premium"].get(r.customer_id, False)) if r.customer_id in seg.index else False
        if r.status in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
            rt, pr = "human_review", "NORMAL"
        elif r.category == "fraud":
            rt, pr = "security_team", "HIGH"
        elif premium:
            rt, pr = "premium_desk", "HIGH"
        else:
            rt, pr = "support_queue", "NORMAL"
        out.append({"ticket_id": r.ticket_id, "customer_id": r.customer_id, "status": r.status,
                    "category": r.category, "segment_name": name, "priority": pr,
                    "route": rt, "tokens_used": r.tokens_used})
    return pd.DataFrame(out)


if __name__ == "__main__":
    tickets = pd.read_csv("data/raw/tickets.csv")
    truth = tickets[["ticket_id", "true_category"]]

    # ---------- redaction report ----------
    counts = {"email": 0, "phone": 0, "order id": 0}
    redacted_all = [redact("" if pd.isna(x) else str(x), counts) for x in tickets["text"]]
    print("=== redaction (all 164 tickets) ===")
    print(counts, "total:", sum(counts.values()), "(reference 55 / 91 / 132, total 278)")
    left = sum(len(re.findall(p, x)) for x in redacted_all for _, p, _ in PATTERNS)
    print("matches left after re-applying the three patterns:", left, "(reference 0)")
    assert left == 0

    # ---------- guardrail counts ----------
    empty = int((tickets["text"].fillna("").str.strip() == "").sum())
    long_ = int(tickets["text"].fillna("").map(mock_llm.est_tokens).gt(MAX_INPUT_TOKENS).sum())
    print(f"empty tickets: {empty}   long tickets: {long_}   (reference 4 and 4)")

    # ---------- full runs, no budget ----------
    print("\n=== zero-shot full run (retry log) ===")
    zero, zlog, zsent, ztok = run_pipeline(tickets, ZERO_SHOT, None, show_retry_log=True)
    few, flog, fsent, ftok = run_pipeline(tickets, FEW_SHOT, None)

    for sent in (zsent, fsent):                          # no personal data in any prompt sent
        for p in sent:
            assert not any(re.search(pat, p) for _, pat, _ in PATTERNS), "personal data reached the model"
    print("assertion passed: no email, phone or order id in any user message sent to the model")
    waits = [w for _, _, w in zlog]
    print(f"retries: {len(zlog)} (reference 19); wait range {min(waits):.4f} to {max(waits):.4f} s "
          f"(allowed 0.045 to 0.0575)")
    assert all(0.045 <= w <= 0.0575 for w in waits)

    print("\nstatus counts, zero-shot:", zero["status"].value_counts().to_dict(),
          "(reference 135 / 12 / 9, plus 4 empty and 4 long)")

    print("\n=== zero-shot versus few-shot ===")
    for name, res, tok in [("zero-shot", zero, ztok), ("few-shot", few, ftok)]:
        n, correct = accuracy(res, truth)
        print(f"{name:<10} classified {n}  accuracy {correct / n * 100:.1f}% ({correct} of {n})  "
              f"total tokens {tok}  tokens per classified ticket {tok / n:.1f}")
    print("(reference: 147 classified, zero-shot 85.7%, few-shot 100%)")

    ok = zero[zero["status"].isin(["classified_json", "classified_keyword"])].merge(truth, on="ticket_id")
    print("\nzero-shot confusion table (rows = true category, columns = predicted):")
    print(pd.crosstab(ok["true_category"], ok["category"]).to_string())

    # ---------- budget demo ----------
    os.makedirs("outputs", exist_ok=True)
    demo, _, _, demo_tokens = run_pipeline(tickets, ZERO_SHOT, token_budget=6000)
    demo.to_csv("outputs/triage_budget_demo.csv", index=False)
    print(f"\n=== budget demo: zero-shot, TOKEN_BUDGET = 6000 ===")
    print("budget_exhausted tickets:", int((demo["status"] == "budget_exhausted").sum()),
          " tokens used:", demo_tokens)

    # ---------- final run: few-shot, budget 100000, with routing ----------
    final, _, _, _ = run_pipeline(tickets, FEW_SHOT, token_budget=100000)
    segments = pd.read_csv("outputs/customer_segments.csv")
    routed = route(final, segments)
    routed.to_csv("outputs/triage_results.csv", index=False)
    print("\n=== final routing (few-shot, budget 100000) ===")
    print("rows:", len(routed))
    print(routed["route"].value_counts().to_string())
    print(routed["priority"].value_counts().to_string())
