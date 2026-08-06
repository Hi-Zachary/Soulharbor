# SoulHarbor-MH-LongMemEval-30

Chinese, mental-health-adjacent, question-centric long-term conversational memory benchmark.

## Files

- `soulharbor_mh_longmemeval_30.json`: main benchmark as a JSON array.
- `soulharbor_mh_longmemeval_30.jsonl`: the same 30 instances in JSONL.
- `soulharbor_mh_longmemeval_30_oracle.json`: evidence-session-only oracle histories.
- `soulharbor_mh_longmemeval_30_profile_gold.jsonl`: optional SoulHarbor profile-maintenance gold labels.
- `soulharbor_mh_longmemeval_30_report.json`: structural validation and checksums.
- `soulharbor_mh_longmemeval_30_EVALUATION_GUIDE.md`: evaluator migration notes.

## Design

- 30 independent question instances; each question owns one timestamped long history.
- 1070 sessions in total; 34-38 sessions per instance, average 35.667.
- 2504 total user/assistant turns, average 83.467 turns per instance.
- Chinese natural dialogue about study, work, relationships, caregiving, adjustment, grief, performance pressure, health routines, career uncertainty and overcommitment.
- No self-harm content and no synthetic clinical diagnosis.
- No multiple-choice questions.
- No dialogue statements that explicitly label a memory as transient, stable, stale, or profile-worthy.
- Evidence sessions are not concentrated at the end of the history.

## Ability mix

```json
{
  "single-session-user": 5,
  "single-session-assistant": 4,
  "abstention": 3,
  "temporal-reasoning": 4,
  "single-session-preference": 4,
  "multi-session": 5,
  "knowledge-update": 5
}
```

Abstention follows LongMemEval's naming convention: the `question_id` ends in `_abs`. The additional `capability` field explicitly records `abstention` for local aggregation.

## LongMemEval-compatible fields

- `question_id`
- `question_type`
- `question`
- `answer`
- `question_date`
- `haystack_session_ids`
- `haystack_dates`
- `haystack_sessions`
- `answer_session_ids`
- evidence turns contain `has_answer: true`

## SoulHarbor extensions

- `history_id`
- `capability`
- `memory_target`
- `answerable`
- `aliases`
- `evidence_message_ids`
- `superseded_message_ids`
- `evaluation`
- `answer_fields` for structured multi-session questions

## Important ingestion rule

`has_answer`, `message_id`, and all gold fields are evaluator metadata. Strip them before sending turns to the assistant model. Keep an internal mapping from source `message_id` to the database message ID for retrieval evaluation.

## Validation

```json
{
  "schema_version": "1.0-longmemeval-compatible",
  "instance_count": 30,
  "question_type_counts": {
    "single-session-user": 5,
    "single-session-assistant": 4,
    "abstention": 3,
    "temporal-reasoning": 4,
    "single-session-preference": 4,
    "multi-session": 5,
    "knowledge-update": 5
  },
  "category_counts": {
    "academic_stress": 3,
    "workplace_burnout": 3,
    "relationships_boundaries": 3,
    "adjustment_loneliness": 3,
    "caregiving_stress": 3,
    "grief_and_change": 3,
    "performance_anxiety": 3,
    "health_routine": 3,
    "career_uncertainty": 3,
    "creative_overload": 3
  },
  "session_count_total": 1070,
  "sessions_per_instance": {
    "min": 34,
    "max": 38,
    "avg": 35.667
  },
  "message_count_total": 2504,
  "messages_per_instance_avg": 83.467,
  "evidence_messages_per_question_avg": 1.533,
  "mean_evidence_session_relative_position": 0.42,
  "all_questions_open_ended": true,
  "all_dates_sorted": true,
  "all_question_dates_after_history": true,
  "all_evidence_ids_valid": true,
  "explicit_label_language_removed": true,
  "whole_session_duplicate_count": 6,
  "sha256_jsonl": "6338e4c2c20cc5dd050b7f575c6fe4b3a316540afc8f65fb108ce9fe3689e534",
  "sha256_json": "a3999d65678605f0b820f0227c3ad69affbb160f4caae83d770448241b567cd5",
  "sha256_oracle": "e03507c4fc5765c2db1d303d7a19887560a8cf6b593b26ef18d528cb46efcc06",
  "sha256_profile_gold": "e963d0fbb0b301dada0590af6743f09cf6902b96dc49b886a5bb47c86b613be7"
}
```
