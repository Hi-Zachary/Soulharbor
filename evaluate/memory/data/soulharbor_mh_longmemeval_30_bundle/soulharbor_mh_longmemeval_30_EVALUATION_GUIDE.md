# Evaluating SoulHarbor on SoulHarbor-MH-LongMemEval-30

## 1. Load one question instance at a time

Each record already contains the complete history for one question. Create a fresh database or isolated user namespace for every `question_id`.

```python
for item in dataset:
    engine = make_clean_engine(question_id=item["question_id"])
    ingest_history(engine, item)
    retrieval = engine.build_context_with_details(
        user_id=1,
        conversation_id=0,
        current_user_message=item["question"],
        recent_messages=[],
        conversation_summary=None,
        exclude_message_ids=set(),
    )
    hypothesis = reader_answer(item["question"], retrieval.context)
```

Do not share a database between question instances. This follows LongMemEval's question-centric setup and prevents facts from one synthetic user leaking into another.

## 2. Use the real timestamps

Parse `haystack_dates` and assign the same timestamp to every turn in the corresponding session. Do not convert session order into arbitrary week numbers.

```python
from datetime import datetime

def parse_lme_date(value: str) -> int:
    return int(datetime.strptime(value, "%Y/%m/%d (%a) %H:%M").timestamp())
```

## 3. Never expose annotation fields

Before ingesting a turn, send only `role` and `content` to the memory system. Do not include:

- `has_answer`
- `message_id`
- `answer_session_ids`
- `evidence_message_ids`
- `superseded_message_ids`
- `answer`, `aliases`, or `evaluation`

Keep `message_id` only in the evaluator's mapping table.

## 4. Preserve source-to-database message mappings

```python
source_to_db: dict[str, int] = {}
db_to_source: dict[int, str] = {}

for source_turn in session:
    db_id = allocate_integer_message_id()
    source_to_db[source_turn["message_id"]] = db_id
    db_to_source[db_id] = source_turn["message_id"]
    ingest(role=source_turn["role"], content=source_turn["content"], message_id=db_id)
```

The retrieval API should return selected database message IDs or anchor IDs. Convert them back to source IDs before scoring.

## 5. Output format

Keep a LongMemEval-compatible prediction file:

```json
{"question_id":"sh_mh_lme_001","hypothesis":"模型回答"}
```

For diagnostics, write a second JSONL file:

```json
{
  "question_id": "sh_mh_lme_001",
  "hypothesis": "模型回答",
  "retrieved_message_ids": ["..."],
  "retrieved_session_ids": ["..."],
  "active_profiles": ["..."],
  "retrieval_trace": {},
  "latency_ms": 0
}
```

## 6. Retrieval metrics

Calculate retrieval metrics only for answerable questions.

```python
gold_messages = set(item["evidence_message_ids"])
retrieved_messages = set(result["retrieved_message_ids"])
message_recall = len(gold_messages & retrieved_messages) / len(gold_messages)
all_evidence = gold_messages <= retrieved_messages

gold_sessions = set(item["answer_session_ids"])
retrieved_sessions = set(result["retrieved_session_ids"])
session_recall = len(gold_sessions & retrieved_sessions) / len(gold_sessions)
```

Report:

- Message Recall@K
- Session Recall@K
- Any-evidence Recall
- All-evidence Recall
- MRR for single-evidence questions

For `knowledge-update`, also report:

```python
old = set(item["superseded_message_ids"])
stale_only = bool(old & retrieved_messages) and not bool(gold_messages & retrieved_messages)
```

Retrieving both old and current evidence is not automatically an error. `stale_only` is the dangerous case.

## 7. QA scoring by evaluation type

Do not send every answer directly to one generic judge.

### semantic_short_answer

1. normalize punctuation and whitespace;
2. check `answer` and `aliases`;
3. call a semantic judge only if deterministic matching fails.

### temporal_order

Parse or judge the two ordered events using `evaluation.first` and `evaluation.second`.

### knowledge_update

The answer must state the current value. Mentioning only a superseded value is incorrect. Mentioning the old value as historical context is allowed if the current value is clear.

### structured_fields

Score every key in `answer_fields` independently and also report exact all-fields accuracy.

### abstention

Accept an explicit statement that the history does not provide enough information. Reject invented concrete answers.

## 8. Oracle run

Run the same reader on `soulharbor_mh_longmemeval_30_oracle.json`.

- Oracle wrong: likely a question/gold/reader/judge problem.
- Oracle right, full system wrong, evidence absent: retrieval problem.
- Oracle right, evidence retrieved, full system wrong: reader or context-formatting problem.

## 9. Suggested report

```json
{
  "qa": {
    "overall_accuracy": 0.0,
    "by_capability": {},
    "by_memory_target": {},
    "structured_field_accuracy": 0.0,
    "abstention_accuracy": 0.0
  },
  "retrieval": {
    "message_recall_at_k": 0.0,
    "session_recall_at_k": 0.0,
    "all_evidence_recall": 0.0,
    "stale_only_rate": 0.0
  },
  "oracle": {
    "qa_accuracy": 0.0
  },
  "profile": {
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "stale_capture_rate": 0.0,
    "third_party_capture_rate": 0.0,
    "transient_capture_rate": 0.0
  }
}
```

## 10. Profile evaluation

Use the separate profile gold file. Compare only active atomic profiles from the profile store. Do not mix episodic chunks into the predicted profile set.

Use one-to-one semantic matching between predicted profiles and `active_facts`. Separately test whether predicted profiles match `superseded_facts`, `third_party_facts`, or the messages listed under `transient_message_ids`.
