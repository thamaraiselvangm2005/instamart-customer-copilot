# Instamart Customer Intelligence and Support Copilot

## Overview

This project works on the data of an online grocery delivery business (Instamart-style). It has four parts:

- **Part A:** audit and clean the raw data, store it in SQLite, and answer business questions with SQL and pandas.
- **Part B:** customer segments and a lapse-prediction model.
- **Part C:** support-ticket triage.
- **Part D:** policy question answering and a refund agent.

All language-model calls use the supplied offline file `mock_llm.py`, so no API key, account or internet connection is needed. All amounts are in INR.

## Reproduce

Run these commands in order from the repository folder (Python 3.11 or newer).

1. `python -m venv .venv`
2. Activate it: `.venv\Scripts\activate` on Windows, or `source .venv/bin/activate` on macOS/Linux
3. `pip install -r requirements.txt`
4. `python generate_data.py`
5. `python src/part_a_audit.py`
6. `python src/part_a_clean_customers.py`
7. `python src/part_a_clean_orders.py`
8. `python src/part_a_store_validate.py`
9. `python src/part_a_sql.py`
10. `python src/part_a_pandas.py`
11. `python src/part_b_rfm.py`
12. `python src/part_b_kmeans.py`
13. `python src/part_b_segments.py`
14. `python src/part_b_lapse_data.py`
15. `python src/part_b_train.py`
16. `python src/part_b_finish.py`
17. `python src/part_c_triage.py`
18. `python src/part_d_policy_rag.py`
19. `python src/part_d_agent.py`

Every script prints its results next to the reference values. Steps 13, 16, 17 and 19 write the files in `outputs/`, and step 19 also writes the checkpoints in `state/`.

## Part A

### Audit of the raw data

| Check | Value |
| --- | --- |
| Raw orders rows | 13,084 |
| Exact duplicate rows | 119 |
| Rows with status pending | 656 |
| Distinct order_id | 12,309 |
| order_amount not convertible to a number | 120 |
| Negative delivery_minutes | 106 |
| Blank delivery_minutes | 2,541 |
| Customers: blank age | 61 |
| Customers: age outside 10-100 | 6 |
| Customers: distinct raw city spellings | 24 |
| Customers: distinct non-blank gender spellings | 10 |
| Customers: blank gender | 30 |

Shapes: orders (13,084, 9), customers (1,200, 5), products (40, 4). Missing values: orders has 53 blank order_amount (0.41%) and 2,541 blank delivery_minutes (19.42%); customers has 61 blank age (5.08%) and 30 blank gender (2.50%); products has none.

### Cleaning rules

**Customers**

| Rule | Values changed |
| --- | --- |
| City: trim spaces and apply title case (24 spellings become 6 cities) | 476 |
| Gender: map every spelling to male, female or unknown (blank becomes unknown) | 950 (including 30 blank) |
| Age: treat ages outside 10-100 as missing, then fill all missing ages with the median age of the customer's city | 67 filled (61 blank + 6 out of range) |
| signup_date converted to a datetime | all 1,200 |

**Orders** (applied in this order)

| Rule | Rows or values changed |
| --- | --- |
| 1. Convert order_ts to a datetime | all 13,084 rows |
| 2. Remove exact duplicate rows | 119 removed |
| 3. Keep only the record with the latest order_ts for each order_id | 656 older rows removed |
| 4. Convert order_amount to numeric; rebuild missing amounts as round(price_inr x quantity x (1 - discount_pct / 100), 2) | 112 amounts rebuilt |
| 5a. Set negative delivery_minutes to missing | 105 set to missing |
| 5b. Completed orders: fill missing delivery_minutes with the median of the product category among completed orders | 774 filled |
| 5c. Cancelled orders: leave delivery_minutes missing (a cancelled order was never delivered) | 1,199 left missing |

Result: 12,309 orders (11,110 completed and 1,199 cancelled), no pending rows, 0 missing order_amount, and completed revenue of INR 2,998,668.80.

### Storage and validation

The clean tables are saved as `data/clean/customers.csv`, `data/clean/products.csv` and `data/clean/orders.csv`, and in the SQLite database `data/instamart.db` (tables customers, products, orders; order_ts stored as text YYYY-MM-DD HH:MM:SS). `src/part_a_store_validate.py` checks 13 reference points and prints `ALL CHECKPOINTS PASSED`.

### SQL analysis

The seven queries are in `sql/queries.sql` and are run from Python by `src/part_a_sql.py`.

| Query | Result |
| --- | --- |
| Q1 WHERE then HAVING | 30 customers |
| Q2 three-table join | 30 rows; revenues add up to INR 2,998,668.80; largest cell Bengaluru / Electronics, INR 233,909.45 |
| Q3 scalar subquery | 11 products above the average price |
| Q4 LEFT JOIN | 25 customers with no completed order |
| Q5 common table expressions | 550 customers above the average total of INR 2,552.06 |
| Q6 ranking window functions | Top customer per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows where RANK differs from ROW_NUMBER |
| Q7 running total | 183 daily rows; last cumulative value INR 2,998,668.80 |

RANK and ROW_NUMBER differ here because customers with the same spend in a city share the same RANK and the next RANK skips ahead, while ROW_NUMBER always counts 1, 2, 3 without ties.

### pandas cross-check

The pivot table (`pd.pivot_table` with `margins=True`) gives a grand total of INR 2,998,668.80, equal to the SQL total.

| City | Sum (INR) | Mean (INR) | Count |
| --- | --- | --- | --- |
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Statistics of completed order_amount: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, IQR 232.20, fences -283.50 and 645.30, 0 outliers below and 897 above.

Correlation matrix (completed orders):

| | order_amount | discount_pct | quantity | delivery_minutes |
| --- | --- | --- | --- | --- |
| order_amount | 1.000 | -0.095 | 0.087 | -0.003 |
| discount_pct | -0.095 | 1.000 | -0.000 | -0.010 |
| quantity | 0.087 | -0.000 | 1.000 | -0.013 |
| delivery_minutes | -0.003 | -0.010 | -0.013 | 1.000 |

### Exploratory analysis

An online grocery business shared data on 12,309 orders from 1,200 customers in 6 cities, covering 40 products in 5 categories. After cleaning the data, 11,110 orders were completed and 1,199 were cancelled, so we studied only the completed ones.

Order amounts are right-skewed (skewness 4.61). The average order is INR 269.91, but the middle order (the median) is only INR 135.00. This happens because most orders are small, while a few very big orders pull the average up. The middle half of all orders falls between INR 64.80 and INR 297.00.

Using the 1.5 × IQR rule, any order above INR 645.30 is an outlier, which means it is much bigger than normal. There are 897 such orders, and none fall below the lower fence. They are probably real purchases, like costly products or lots of items, and not mistakes.

We also checked whether order amount, discount, quantity and delivery time are connected. All the links were very weak. The strongest one, between order amount and discount, was only about -0.095, which is almost nothing. A correlation this small does not establish causation.

## Part B

### RFM table

Snapshot date 2025-10-01. Only completed orders count. For every customer: recency = days since the last completed order, frequency = number of distinct completed orders, monetary = total order_amount.

| Measure | Value |
| --- | --- |
| Customers with at least one completed order | 1,175 |
| Recency: median / min / max (days) | 19 / 0 / 180 |
| Frequency: mean / max | 9.46 / 33 |
| Monetary: mean / max (INR) | 2,552.06 / 14,766.55 |

### K-means (K = 2 to 8)

The three RFM columns were standardised first (`StandardScaler`), then K-means was run with `n_init=10` and `random_state=42`.

| K | Inertia | Silhouette |
| --- | --- | --- |
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Choice of K = 4.** The elbow is the spot where the line on the graph stops dropping fast and starts going flat, like a bent arm, and for us that spot is K = 4, where inertia is 743.1. I did not pick K = 3 or K = 5 because K = 4 had the best silhouette score (0.526), which means its groups are the cleanest and most separate.

**Why inertia alone cannot choose K.** With 8 groups, inertia looks great (366.5, compared with 1820.8 at K = 2), but that is only because small groups sit close to their centres, and a shop team would have too many groups to act on. Inertia always gets smaller with more clusters because each point gets a centre closer to it, so a lower number alone does not prove the groups are good.

### Segments

| Segment | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Business name | Recommended action |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 138 | 24.12 | 6.77 | 7,129.70 | Premium Big Spenders | VIP treatment: early access, personal offers, priority support |
| 2 | 419 | 8.77 | 19.40 | 3,410.02 | Loyal Regulars | Reward loyalty: points, subscriptions, bundles to raise basket size |
| 0 | 436 | 28.86 | 3.77 | 1,057.62 | Casual Shoppers | Nudge to order more often: free-delivery threshold, category offers |
| 3 | 182 | 109.04 | 2.20 | 685.99 | Dormant Customers | Win-back campaign: reminder message plus a comeback discount |

The cluster numbers are only labels and have no order, so the names were chosen from the numbers (highest mean monetary = Premium, longest recency = Dormant, highest frequency among the rest = Loyal). PCA on the scaled RFM matrix keeps 63.4% and 21.8% of the variance in its first two components, 85.2% in total. The result is saved as `outputs/customer_segments.csv` (1,175 rows; 138 Premium customers).

### Lapse prediction

- **Problem:** by 2025-09-01, which customers will place no completed order in September 2025 (1 to 30 September)?
- **Cohort:** customers with at least one completed order before 2025-09-01: 1,156 customers, of whom 443 lapsed (38.32%).
- **Features** (only orders before the cutoff): recency, frequency, monetary, average discount, cancellation rate, average delivery minutes, age, tenure in days, favourite category, city and gender.
- **Split:** stratified 80/20 with `random_state=42`: 924 training rows (354 lapsed, 570 not lapsed) and 232 test rows.
- **Baseline** "nobody lapses" on the test set: accuracy 61.6%, F1 of the lapsed class 0.00.

**No leakage.** Features use only orders dated before 2025-09-01, and the label comes from September. The scaler and the one-hot encoder sit inside a scikit-learn Pipeline, so they are fitted on training rows only; the cross-validation folds are refitted inside the pipeline as well. The scaler's frequency mean is 8.115 (training rows) while the mean of the whole cohort is 8.029: they differ because the test rows must never influence the scaler. The segment labels were not used as features.

**Tuning** (5-fold cross-validated F1 of the lapsed class, training data only):

| Model | Best setting | CV F1 |
| --- | --- | --- |
| Logistic regression | C = 100 | 0.651 |
| Random forest (100 trees) | max_depth = 8, min_samples_leaf = 10 | 0.687 |

**Test-set results** (lapsed = positive class; confusion matrix is [[TN, FP], [FN, TP]]):

| Model | Confusion matrix | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Logistic regression | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | [[111, 32], [32, 57]] | 0.640 | 0.640 | 0.640 |

**Selected model: random forest.** We picked the model using the cross-validation score, which comes from the training data only. The random forest scored 0.687 and logistic regression scored 0.651. We did not use the test score to choose, because the test set must stay untouched until the end. On the test set the two models were almost the same (F1 0.640 for the forest and 0.648 for logistic regression), and both are far above the 0.55 minimum and the baseline of 0. For a retention team, missing a customer who will leave costs more than sending an offer to a customer who would have stayed, so finding lapsing customers (recall) matters most. The forest also handles non-linear patterns, which logistic regression cannot.

**Class-balance experiment.** The lapsed training rows were randomly copied (with `random_state=42`) until both classes had 570 rows. Only the training set was changed; the test set is the original one.

| Logistic regression (C = 100) | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Original training data | 0.644 | 0.652 | 0.648 |
| Balanced training data | 0.589 | 0.854 | 0.697 |

Recall changed most (0.652 to 0.854) and precision fell (0.644 to 0.589). The balanced model finds many more lapsing customers but also flags more customers who would have stayed.

**Interpretation.** The five features that matter most for the random forest are monetary (0.278), frequency (0.277), recency (0.111), average delivery minutes (0.062) and average discount (0.053). In logistic regression, recency has a small positive coefficient (+0.040), so customers who bought longer ago are a bit more likely to lapse. Frequency has a large negative coefficient (-1.698), so customers who order often tend to stay. Cancellation rate has a very small effect (-0.124).

**Saved model and retention list.** The random forest is saved as `outputs/lapse_model.joblib`; reloading it gives exactly the same predictions on the test set. `outputs/retention_list.csv` holds the ten Premium test-set customers with the highest predicted lapse probability.

| customer_id | lapse_probability | actually_lapsed |
| --- | --- | --- |
| C0862 | 0.5304 | 0 |
| C0060 | 0.4765 | 0 |
| C1008 | 0.4756 | 0 |
| C0267 | 0.4753 | 0 |
| C0133 | 0.4712 | 0 |
| C0942 | 0.4688 | 0 |
| C1000 | 0.4654 | 0 |
| C0553 | 0.4645 | 0 |
| C1112 | 0.4620 | 0 |
| C0565 | 0.4603 | 1 |

Only 1 of these 10 customers truly lapsed. This list is illustrative and is not a leakage-free evaluation, because the Premium flag comes from RFM numbers calculated over the whole period, which includes September.

## Part C

All model calls go through the offline `mock_llm.py`. Tickets are processed in file order.

### Guardrails and privacy

| Step | Result |
| --- | --- |
| Empty tickets (skipped, `skipped_empty`) | 4 |
| Tickets above 400 estimated tokens (`flagged_long`) | 4 |
| Redaction (applied in this order: email, phone, order id) | 55 emails, 91 phones, 132 order ids, 278 in total |
| Matches left after applying the three patterns again | 0 |

Each ticket is redacted before it is sent, so the model only sees placeholders such as `[EMAIL]`, `[PHONE]` and `[ORDER_ID]`. The script asserts that no email, phone number or order id appears in any message sent to the model.

### Prompts, retries and validation

The zero-shot prompt asks for one of billing, delivery, technical or fraud and for JSON only. The few-shot prompt has the same instruction plus a line `EXAMPLES:` with three labelled examples, including one fraud example.

Transient errors are retried up to 3 times with exponential backoff and a random factor: wait = 0.05 x 2^retry x uniform(0.9, 1.15) seconds. In a full run there were 19 retries, with waits between 0.0481 s and 0.0573 s, and every ticket succeeded after at most one retry.

Every answer is validated. Valid JSON with an allowed category is `classified_json`; if the text is not JSON but contains exactly one allowed category word it is `classified_keyword`; everything else (including valid JSON with an unknown category) is `flagged_invalid_output` and never guessed.

Status counts of a full zero-shot run: 135 `classified_json`, 12 `classified_keyword`, 9 `flagged_invalid_output`, 4 `skipped_empty` and 4 `flagged_long` (164 tickets).

### Zero-shot against few-shot

| Prompt | Classified | Accuracy | Total tokens | Tokens per classified ticket |
| --- | --- | --- | --- | --- |
| Zero-shot | 147 | 85.7% (126 of 147) | 8,308 | 56.5 |
| Few-shot | 147 | 100.0% (147 of 147) | 17,200 | 117.0 |

Confusion table of the zero-shot run (rows = true category, columns = predicted):

| | billing | delivery | fraud | technical |
| --- | --- | --- | --- | --- |
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

**Is few-shot worth the cost?** Few-shot costs about twice as many tokens for each ticket (117.0 against 56.5), but it got every ticket right (100.0%, against 85.7% for zero-shot). The zero-shot mistakes were 11 fraud tickets and 10 technical tickets that were read as billing. A fraud ticket that waits in the billing queue can cause real harm, so the extra tokens are worth paying when a mistake is costly. When tickets are low risk and there are very many of them, zero-shot is cheaper and may be good enough.

### Token budget

With `TOKEN_BUDGET = 6000` and the zero-shot prompt, 43 tickets were marked `budget_exhausted` (6,012 tokens were used before the check stopped further calls). The result is saved in `outputs/triage_budget_demo.csv`.

### Routing

The final run (few-shot, budget 100,000) is saved in `outputs/triage_results.csv`. Rules: any ticket that was skipped, too long, invalid or budget-exhausted goes to `human_review`; a `fraud` ticket goes to `security_team` with priority HIGH; any other ticket of a Premium customer goes to `premium_desk` with priority HIGH; everything else goes to `support_queue`.

| Route | Tickets |
| --- | --- |
| support_queue | 102 |
| security_team | 34 |
| human_review | 17 |
| premium_desk | 11 |

Priorities: 45 HIGH and 119 NORMAL. Tickets classified by the model and accepted by the validation rules were routed automatically; the 17 `human_review` tickets are the ones that need a person.

## Part D

### Policy documents, chunking and retrieval

The five policy files are split into chunks of 40 words that overlap by 10 words (step 30). This gives 15 chunks: refund 4, delivery 3, escalation 3, payment 3 and privacy 2. Retrieval uses `TfidfVectorizer` and cosine similarity; the question is transformed with the same fitted vectorizer. For the query "refund policy late delivery" all three best chunks come from `refund_policy.txt`. The answer step sends the three best chunks as `CONTEXT:` and the question as `QUESTION:` to the mock model.

**Evaluation** (12 answerable and 2 unanswerable questions):

| Setting | Chunks | Right document in the top 3 | Expected fact in the answer | Correct "could not find" |
| --- | --- | --- | --- | --- |
| Overlap 10 | 15 | 12 of 12 | 12 of 12 | 2 of 2 |
| Zero overlap | 13 | 12 of 12 | 10 of 12 | 2 of 2 |

With zero overlap two questions fail: "What delivery time makes a completed order eligible for a full refund?" (the expected "more than 45 minutes") and "How soon is money returned when a payment fails but is deducted?" (the expected "48 hours"). At zero overlap the sentence that holds each answer is cut between two chunks, and the best-matching chunk has only half of it (for example "receives a full refund of the order amount" without "more than 45 minutes"), so the fact is missing from the answer. The 10-word overlap keeps the sentence together.

**TF-IDF in one paragraph.** TF (term frequency) counts how often a word appears in a chunk. IDF (inverse document frequency) checks how rare the word is across all the chunks, so a word that appears in every chunk counts for little, and a rare word counts for more. A chunk scores high when it contains the rare words from the question. The weakness is that TF-IDF only matches exact words, so it does not understand that "money back" and "refund" mean the same thing.

### The refund agent

The agent has five tools: `get_order` (reads the SQLite database), `search_policy` (uses the TF-IDF retrieval), `calculate_refund` (full order amount if delivery took more than 45 minutes, otherwise 0), `issue_refund` (refunds up to INR 1000 and writes a line to `outputs/refund_ledger.jsonl`) and `escalate_to_human` (for refunds above INR 1000, also written to the ledger). It follows the ReAct pattern: the model chooses a tool, the code runs it, the result goes back to the model, and this repeats until the model gives a final answer. Limits: at most 6 model calls, 6,000 tokens and 10 seconds. Every run is logged in `outputs/agent_runs.jsonl`.

| Scenario | Status | Refund (INR) | Model calls | Tokens |
| --- | --- | --- | --- | --- |
| S1 | REFUND | 128.25 | 5 | 1,042 |
| S2 | REFUND | 135.00 | 5 | 1,042 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,029 |
| S4 | DENIED | 0.00 | 4 | 721 |
| S5 | DENIED | 0.00 | 3 | 424 |
| S6 | NOT_FOUND | 0.00 | 2 | 130 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,174 |

All seven results match the expected outcomes. S7 keeps searching the policy and never finishes, so it stops after 6 calls and hands over to a human.

**Stopping safely.** With `token_limit=200` the run on S1 stops with `FALLBACK_TOKEN_LIMIT` after 3 calls (441 tokens) without running the tool it had just requested. With `time_limit_seconds=0` it stops with `FALLBACK_TIMEOUT` before any model call. Each fallback returns a short hand-off summary for a human (status, number of calls and steps, and the last tool used).

**Crash and resume.** After every tool result the full state (messages, tokens, steps, tools executed) is saved to `state/<run_id>.json`. In the demonstration S1 crashes after its second tool; `resume_agent` loads the checkpoint and continues. The tools executed over both parts are `get_order`, `search_policy`, `calculate_refund` and `issue_refund`, each exactly once, and the final result is REFUND of INR 128.25. The demonstration writes to its own ledger (`state/crash_demo_ledger.jsonl`), which holds exactly one line, so the refund was not issued twice.

**ReAct or plan-and-execute for this agent.** In ReAct the agent takes one step, looks at the result, and then decides the next step. This suits the refund agent, because every step depends on the one before it. For example, if the order does not exist (S6) or was cancelled (S5), the agent can stop early instead of doing steps it had planned. In plan-and-execute the agent writes the whole plan first and then runs it. That uses fewer model calls and is easier to check, but the plan is made before the agent has seen the order, so some steps may turn out to be useless or wrong. For refunds, ReAct is the better choice, as long as limits stop it from looping forever, like S7, which kept searching until it reached the limit of 6 calls.

## Limitations and Next Steps

- All data is synthetic and the model calls use an offline mock, so the numbers show that the pipeline works but say nothing about real customers or a real language model.
- The lapse model uses one cutoff date and one random split. A time-based validation over several cutoffs would be more reliable.
- The Premium flag is computed from RFM over the whole period, which includes September, so the retention list is illustrative and not a leakage-free evaluation (only 1 of the 10 listed customers truly lapsed).
- TF-IDF retrieval only matches words; embeddings would also find paraphrases. The refund rule (a 45-minute threshold and an INR 1000 limit) is deliberately simple.
- Next steps: add monitoring of the triage accuracy on real tickets, calibrate the lapse probabilities, and add a human approval step for escalated refunds.
