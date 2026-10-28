---
layout: post
title: "Structured Call Summaries: Turning Long, Noisy Transcripts into Clean JSON with an LLM"
summary: "How one LLM call per conversation, constrained by a prompt and a JSON schema, turns about 820,000 noisy transcripts a day into clean, structured summaries."
project: call-intelligence
date: 2026-10-28 09:01:00
---

*Part 2 of a 4-part series on Call Intelligence. [Part 1](/posts/2026/10/28/call-intelligence-1-pipeline/) covered the end-to-end pipeline. This part focuses on the call summary model, the step everything else depends on.*

> All transcripts, names, amounts, IDs and screens in this article are **mock data** created for illustration.

---

## The problem with transcripts

A single call transcript can run to **thousands of words**. Because it comes from speech-to-text, it's also full of errors: misheard names, broken sentences, numbers spelled out as words, and long stretches of hold messages and transfer chatter.

At about 300 million calls a year, that's roughly **820,000 of these transcripts every day**. Using them directly is painful for everyone downstream:

- **Manual review** is slow. A reviewer has to read the whole call to find the part that mattered.
- **Analytics** can't count free-form dialogue.
- **Other models** (topics, entities) spend tokens and accuracy on noise.

We needed one step that **cleans every conversation into the same structure** before any person or model uses it.

<figure class="diagram">
{% include diagrams/ci2-foundation.svg %}
<figcaption>Figure 1: The structured summary is the shared input for the topic model, entity matching and human review.</figcaption>
</figure>

---

## The approach: zero-shot, schema-constrained summarization

We didn't train or fine-tune a model. The summary model is **zero-shot prompting** against a fast, low-cost LLM, with two controls:

1. **A prompt that defines every section** and enforces strict factuality.
2. **A response JSON schema** passed to the API, so the model *must* return the target structure.

The prompt tells the model what to write. The schema guarantees the shape of what comes back. At this volume the shape matters as much as the content: one malformed response in a thousand is still hundreds of broken rows a day.

<figure class="diagram">
{% include diagrams/ci2-pipeline.svg %}
<figcaption>Figure 2: Daily batch flow for the summary model.</figcaption>
</figure>

---

## Step 1: Prepare the transcript

Transcripts land in BigQuery as one row per turn (see Part 1). Before prompting, the turns for each call are put back in order and labeled by speaker, so the model reads one clean conversation:

```sql
SELECT
  call_id,
  STRING_AGG(CONCAT(speaker, ': ', utterance_text), '\n' ORDER BY turn_seq) AS transcript
FROM contact_intel.call_transcript
WHERE call_date = @run_date
  AND TRIM(utterance_text) != ''
GROUP BY call_id;
```

The batch runs on calls from **a few days earlier**, once their transcripts are complete.

---

## Step 2: Define what a summary contains

<figure class="diagram">
{% include diagrams/ci2-schema.svg %}
<figcaption>Figure 3: Sections of the structured summary.</figcaption>
</figure>

| Section | What it captures | Rules that mattered |
|---|---|---|
| **Concise summary** | Key events and actions by each party | Be specific: include amounts, dates and commitments; capture the caller's experience (frustration, satisfaction); under 200 words |
| **Contact information** | Caller name, who the call is about, caller type, relationship | Caller type is an **enum**: Provider, Member's representative, Member, Not mentioned. Use "Self" when calling for themselves |
| **Call intents & solutions** | Each question or issue, paired with the agent's response | One topic per intent, 20 words or fewer; include specific keywords ("dental" claim, "prior" authorization); split multi-topic intents |
| **Repeated issue** | Has the caller contacted us multiple times about the same unresolved issue? | Integer: 1 or 0 |
| **Call transfers** | Number of transfers during the call | Don't count the final hand-off to a post-call survey |
| **Next steps: caller** | What the caller still has to do | List; empty when none |
| **Next steps: agent** | What the agent committed to | List; empty when none |
| **Future call watchouts** | What an agent should be careful about if the caller calls again | List; empty when none |
| **Entities** | Benefits, specialists, diseases, procedures explicitly mentioned | Avoid generic terms ("coverage", "doctor", "test"); skip transcription errors |

Three of these rules had an outsized effect:

- **Granular, keyword-rich intents.** "Inquire about a claim" is useless for classification. "Inquire about a denied **dental** claim" maps cleanly to a topic. The topic model (Part 3) reads these intents, so intent quality directly drives topic quality.
- **Consistent qualifiers across intents.** If one intent says "vision claim," every related intent should too, even if it feels repetitive. Consistency beats elegance for downstream models.
- **Excluding the survey transfer.** Without this rule, almost every call showed at least one transfer, which made the metric meaningless.

---

## Step 3: Write the prompt

```text
You are a data labeler generating a structured summary of a
health insurance call transcript between:
  - a call center agent (AGENT)
  - a member, a member's representative, or a provider (CALLER)

Your knowledge is STRICTLY LIMITED to the transcript.
Do not make assumptions or predictions beyond it.

Produce ONLY these sections:

Concise_Summary:
  Key events and actions by each party. Be specific: include
  amounts, dates, and commitments. Capture the caller's experience.

Contact_Information:
  Name of caller; name(s) of person(s) the call is about;
  type of caller (Provider | Member's representative | Member |
  Not mentioned); relationship ("Self" if calling for themselves).

Call_Intents_and_Solutions:
  For each intent: one specific topic, 20 words or fewer, with
  relevant keywords. Split multi-topic intents.
  Pair each with the agent's solution.

Repeated_Issue: 1 if evidence of repeat contact, else 0.
Call_Transfers: count transfers, excluding the final survey transfer.
Next_Steps_Caller / Next_Steps_Agent / Future_Call_Watchouts:
  list each item; "None" if there are none.
Entities: benefits, specialists, diseases, procedures explicitly
  mentioned. Avoid generic terms and transcription errors.

TRANSCRIPT:
{transcript}
```

*Simplified.* The first rule does the most work. **"Your knowledge is strictly limited to the transcript"** is the main defense against the model filling gaps with plausible but invented details.

---

## Step 4: Pin the shape with a response schema

The schema is passed in the API call, so every response is machine-readable. An abridged version:

```json
{
  "type": "object",
  "properties": {
    "concise_summary": { "type": "string" },
    "contact_information": {
      "type": "object",
      "properties": {
        "name_of_caller": { "type": "string" },
        "name_of_person_the_call_is_about": { "type": "array", "items": { "type": "string" } },
        "type_of_caller": {
          "type": "array",
          "items": {
            "type": "string",
            "enum": ["Provider", "Member's representative", "Member", "Not mentioned"]
          }
        },
        "relationship_to_person_the_call_is_about": { "type": "array", "items": { "type": "string" } }
      },
      "required": ["name_of_caller", "name_of_person_the_call_is_about",
                   "type_of_caller", "relationship_to_person_the_call_is_about"]
    },
    "call_intents_and_solutions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "intent":   { "type": "string" },
          "solution": { "type": "string" }
        },
        "required": ["intent", "solution"]
      }
    },
    "repeated_issue":        { "type": "integer" },
    "call_transfers":        { "type": "integer" },
    "next_steps_caller":     { "type": "array", "items": { "type": "string" } },
    "next_steps_agent":      { "type": "array", "items": { "type": "string" } },
    "future_call_watchouts": { "type": "array", "items": { "type": "string" } },
    "entities": {
      "type": "object",
      "properties": {
        "benefits":    { "type": "array", "items": { "type": "string" } },
        "specialists": { "type": "array", "items": { "type": "string" } },
        "diseases":    { "type": "array", "items": { "type": "string" } },
        "procedures":  { "type": "array", "items": { "type": "string" } }
      },
      "required": ["benefits", "specialists", "diseases", "procedures"]
    }
  }
}
```

Two schema details worth copying:

- **Enums for categorical fields.** `type_of_caller` can only be one of four values, so it can be counted reliably.
- **Arrays even for "single" fields.** A call can be *about* more than one person (a parent calling for two children), so names and relationships are lists.

---

## Step 5: Call the model

| Setting | Value | Why |
|---|---|---|
| Model | Fast, low-cost LLM (Gemini Flash class) | ~820,000 calls a day; summarization doesn't need the largest model |
| Temperature | 0.1 | Near-deterministic, factual output |
| Top-p | 0.95 | Standard nucleus sampling |
| Retries | Exponential backoff | Absorbs rate limits and transient API errors in large daily batches |

```python
import backoff
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

@backoff.on_exception(backoff.expo,
                      (ResourceExhausted, ServiceUnavailable),
                      max_tries=5)
def summarize(transcript: str) -> dict:
    resp = model.generate_content(
        PROMPT.format(transcript=transcript),
        generation_config={
            "temperature": 0.1,
            "top_p": 0.95,
            "response_mime_type": "application/json",
            "response_schema": RESPONSE_SCHEMA,
        },
    )
    usage = resp.usage_metadata
    return {
        "summary": resp.text,
        "input_token": usage.prompt_token_count,
        "output_token": usage.candidates_token_count,
    }
```

The daily job runs `summarize()` across the day's calls in parallel and writes one row per call. *Simplified:*

```python
from concurrent.futures import ThreadPoolExecutor

def run_summaries(run_date: str) -> None:
    calls = bq.query(TRANSCRIPTS_SQL, job_config=params(run_date)).result()

    def one(call):
        out = summarize(call.transcript)
        return {
            "call_id": call.call_id,
            "call_date": call.call_date,
            "summary": encrypt(out["summary"]),      # field-level encryption
            "input_token": out["input_token"],
            "output_token": out["output_token"],
            "run_date": today(),
        }

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(one, calls))

    write_partition("contact_intel.call_summary", rows, run_date)  # rerun = replace the day
```

---

## Example (mock)

**Transcript excerpt**

```text
AGENT:  Good morning, who am I speaking with?
CALLER: This is Jane Doe, calling for myself.
AGENT:  How can I help today?
CALLER: My thyroid medication refill cost $95 this time.
        Last month it was $30 for the same prescription.
CALLER: I already asked the pharmacist and they couldn't explain it.
AGENT:  I understand. Let me connect you with our pharmacy benefits team.
        [transfer]
...
```

**Structured summary**

```json
{
  "concise_summary": "Jane Doe called about her thyroid medication refill increasing from $30 to $95 for the same prescription. She had already asked the pharmacist, who could not explain the change, and she was confused. The agent transferred her to the pharmacy benefits team.",
  "contact_information": {
    "name_of_caller": "Jane Doe",
    "name_of_person_the_call_is_about": ["Jane Doe"],
    "type_of_caller": ["Member"],
    "relationship_to_person_the_call_is_about": ["Self"]
  },
  "call_intents_and_solutions": [
    {
      "intent": "Understand why thyroid medication copay rose from $30 to $95",
      "solution": "Agent transferred the caller to the pharmacy benefits team"
    }
  ],
  "repeated_issue": 0,
  "call_transfers": 1,
  "next_steps_caller": [],
  "next_steps_agent": [],
  "future_call_watchouts": [
    "Caller already asked the pharmacist; be ready to explain the copay change if she calls again"
  ],
  "entities": {
    "benefits": ["prescription coverage"],
    "specialists": ["pharmacist"],
    "diseases": [],
    "procedures": []
  }
}
```

Note what the model **didn't** do: it didn't guess a diagnosis from "thyroid medication," and it didn't invent next steps the transcript doesn't support.

![Session review](/assets/img/call-intelligence/mock_session_review.png)
*Figure 4: Session Review shows the transcript and decoded summary side by side (mock data).*

---

## Step 6: Store it safely

Each call produces one row:

| Field | Type | Description |
|---|---|---|
| `call_id` | STRING | Unique identifier for the call |
| `call_date` | DATE | Date the call occurred |
| `summary` | STRING | Structured summary JSON (**encrypted at field level**) |
| `input_token` | INT64 | Tokens in the input transcript |
| `output_token` | INT64 | Tokens in the generated summary |
| `run_date` | DATE | Date the model ran for this call |

```sql
CREATE TABLE contact_intel.call_summary (
  call_id      STRING,
  call_date    DATE,
  summary      STRING,   -- encrypted JSON
  input_token  INT64,
  output_token INT64,
  run_date     DATE
)
PARTITION BY call_date
CLUSTER BY call_id;
```

Two design choices:

- **Encrypt the summary field.** A summary contains names, relationships and health details. With field-level encryption, table access alone isn't enough to read it; consumers decrypt only when they need to.
- **Store token counts.** They track cost and verbosity, and they flag outliers such as unusually long calls or a prompt change that suddenly doubles output length.

Token counts make that check a simple daily query:

```sql
SELECT
  call_date,
  COUNT(*)                                        AS calls,
  AVG(input_token)                                AS avg_input_tokens,
  AVG(output_token)                               AS avg_output_tokens,
  APPROX_QUANTILES(output_token, 100)[OFFSET(95)] AS p95_output_tokens
FROM contact_intel.call_summary
WHERE call_date BETWEEN @start_date AND @end_date
GROUP BY call_date
ORDER BY call_date;
```

A jump in `avg_output_tokens` right after a prompt change is the first sign the model has started writing longer, not better.

---

## How we measure quality

Summaries are evaluated by **human review** of a sample of calls, **section by section**, on two metrics:

| Metric | Question | Scale |
|---|---|---|
| **Accuracy** | Of what the summary says, how much is correct? | 0 = Bad, 0.5 = Good enough, 1 = Excellent |
| **Completeness** | Of the important details in the call, how much did the summary capture? | 0 = Missing key info, 0.5 = Partial, 1 = Fully captured |

A simple way to explain it to stakeholders:

- **Accuracy** is the basket test: of all the summaries in the basket, how many truly represent their calls?
- **Completeness** is the puzzle test: of all the pieces in the original call, how many made it into the summary?

Scoring **per section** matters. An overall score hides problems; a section score tells you exactly which prompt instruction to fix. Once reviewers fill in a score sheet, the roll-up is a few lines:

```python
import pandas as pd

# one row per (call_id, section) with reviewer scores of 0, 0.5 or 1
scores = pd.read_csv("review_scores.csv")

by_section = (scores.groupby("section")[["accuracy", "completeness"]]
                    .mean()
                    .sort_values("accuracy"))
overall = scores[["accuracy", "completeness"]].mean()
```

The lowest rows of `by_section` are the prompt instructions to work on next.

<figure class="diagram">
{% include diagrams/ci2-evaluation.svg %}
<figcaption>Figure 5: Human evaluation today, and the planned move to automated evaluation.</figcaption>
</figure>

---

## What's next

Human review doesn't scale to 820,000 summaries a day. The next steps are:

- **LLM-as-judge evaluation:** use a model to score summaries with the same accuracy, completeness and hallucination rubric, so every new prompt or model version can be benchmarked quickly.
- **Continuous monitoring:** score a sample of production summaries regularly to catch drift.
- **Post-response validation:** check the returned JSON for completeness and format, with fallback handling for failures. The response schema prevents most structural problems, but it can't catch an empty intent list or a section filled with placeholder text.

---

## Lessons learned

1. **Constrain the output twice.** The prompt defines the content; the response schema enforces the shape.
2. **Limit the model's knowledge explicitly.** "Only what's in the transcript" is the cheapest hallucination control there is.
3. **Write intents for the next model, not for a human.** Granular, keyword-rich intents are what make topic classification work.
4. **Define edge cases in the prompt.** Small rules, like excluding the survey transfer, decide whether a field is usable.
5. **Use low temperature.** Summaries should be repeatable, not creative.
6. **Treat summaries as sensitive data.** Encrypt at the field level and log token counts.
7. **Score section by section.** It turns "the summary isn't great" into a specific fix.

---

## The rest of the series

- **Part 1:** [How We Turn 300 Million Calls a Year into Actionable Insights](/posts/2026/10/28/call-intelligence-1-pipeline/)
- **Part 3:** [LLM as Teacher, XLNet as Student: Classifying Why Members Call](/posts/2026/10/28/call-intelligence-3-topics/)
- **Part 4:** [Named Entities from Call Summaries: Why We Chose a Dictionary over an LLM](/posts/2026/10/28/call-intelligence-4-entities/)
