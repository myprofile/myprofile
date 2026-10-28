---
layout: post
title: "From Raw Call Transcripts to Contact Intelligence: A Summary-First AI Pipeline on GCP"
summary: "How a summary-first LLM pipeline on GCP turns every raw call transcript into one queryable table of reasons, topics, and entities."
date: 2026-10-28 09:00:00
---

*Part 1 of a 4-part series on turning millions of customer conversations into structured, measurable data.*

> All numbers, labels, and screens in this article are **mock data** created for illustration.

---

## The problem

A contact center records every conversation, but most of that information stays locked in transcripts. Operational data tells you *how many* people called and *how long* the calls took. It doesn't tell you *why* they called, *what* they talked about, or *whether the issue was resolved*.

Traditional approaches have clear limits:

- **Disposition codes** that agents pick by hand are slow, inconsistent, and too broad.
- **Keyword rules on raw transcripts** break on transcription errors and varied phrasing.
- **Manual QA sampling** reviews only a small fraction of calls.

We wanted a pipeline that could read **every** call, turn it into structured data keyed by `call_id`, and combine it with the operational data we already had, in **one table** a dashboard could query quickly.

---

## The key idea: summarize first

Raw transcripts are long and noisy: thousands of words, transcription errors, greetings, identity checks, hold periods. Every model that reads them directly pays for that noise.

So the pipeline is built around one decision: **an LLM turns each transcript into a structured summary first**, and the other models read the summary instead of the transcript.

![Summary-centric design](/assets/img/call-intelligence/03_summary_centric.png)
*Figure 1: The structured summary is the shared input for topics, entities, and human review.*

We also didn't use an LLM for every step. Each layer uses the simplest tool that does the job well:

| Layer | Technique | Why |
|---|---|---|
| **Structured summary** | LLM with a response schema | Conversations are messy; this is where language understanding pays off |
| **Call topics** | XLNet fine-tuned on LLM-labeled data | High-volume classification; the same input always gets the same output |
| **Named entities** | Keyword dictionary with longest-match rule | Fixed categories; explainable and cheap |
| **Resolution** | LLM prompt on the transcript | Needs judgment about how the conversation ended |
| **Operational metrics** | Standard contact-center calculations | Already structured |

---

## Architecture overview

![Architecture](/assets/img/call-intelligence/01_architecture.png)
*Figure 2: End-to-end architecture. Each enrichment pipeline writes its own table. A consolidation pipeline builds one dashboard table after all of them finish.*

There are three layers:

1. **Ingestion:** transcripts land in Cloud Storage and are loaded into BigQuery.
2. **Enrichment:** the summary, resolution, and metrics jobs read the transcripts; the topic and entity jobs read the summaries.
3. **Consolidation:** a final pipeline waits for every upstream job, then writes one partitioned, clustered table that powers the dashboard.

Everything runs as a **daily batch**. Because transcripts and summaries need to be complete first, AI fields land **a few days after the call**, which is fine for trend analysis but worth stating clearly to dashboard users.

---

## Step 1: Landing the transcripts

The transcription platform exports transcripts as files into a Cloud Storage bucket. A scheduled job loads them into BigQuery:

```sql
-- Mock schema
CREATE TABLE contact_intel.call_transcript (
  call_id        STRING,
  member_id      STRING,      -- tokenized / de-identified
  lob            STRING,
  call_date      DATE,
  speaker        STRING,      -- AGENT | CALLER
  turn_seq       INT64,
  utterance_text STRING
);
```

Everything downstream is keyed on **`call_id`**.

---

## Step 2: Structured summaries (LLM)

An LLM reads each transcript and returns a **structured JSON summary**. It doesn't return a paragraph. The prompt defines each section, and a **response schema** passed to the API guarantees the shape of the output.

Sections include:

- **concise summary**
- **contact information** (who called, about whom, caller type)
- **call intents with the agent's solution for each**
- **repeated issue** and **transfer count**
- **next steps** for caller and agent
- **future call watchouts**

The prompt's most important rule is that the model's knowledge is **strictly limited to the transcript**, which is the main defense against invented details. Temperature is kept low (0.1) for repeatable output, and the summary field is **encrypted** at rest because it contains personal and health details.

*Part 2 covers the schema, prompt, and evaluation in depth.*

---

## Step 3: Call topics (fine-tuned XLNet)

The topic model tags each call with topics from a **pre-defined taxonomy**, organized as topic groups and topics. A call can have **several topics**.

It reads the summary's concise summary and call intents, and it works in two stages:

1. **An LLM labels training data.** Few-shot, chain-of-thought prompts first pick a topic group, then a topic, for each intent and for the concise summary.
2. **XLNet learns from those labels.** A pre-trained XLNet model is fine-tuned as a multi-label classifier, with rare and common topics modeled separately. XLNet is what runs daily in production.

Output is one row per call with a **list** of topics:

```json
{
  "call_id": "C-000123",
  "call_date": "2026-03-12",
  "call_topics": ["Ask why a claim was denied", "Confirm a provider is in-network"]
}
```

Calls with no matching topic are reported as **Unfiltered**. They're either empty calls (for example, quick disconnects) or topics the taxonomy doesn't cover yet.

*Part 3 covers the teacher-student design and evaluation.*

---

## Step 4: Named entities (keyword dictionary)

The entity model pulls specific **benefits, specialists, procedures, and diseases** from the summary and maps each one to a standard category.

It doesn't use an LLM. It uses **curated keyword-to-category tables**, one per entity type:

- Keywords are expanded into singular and plural variants.
- Exact matches are found in the summary text.
- When matches overlap, the **longest match wins**, so "obstetric panels" isn't also counted as a generic lab "panel".

Each match keeps both the **value** (the words as they appeared) and the **class** (the standard category):

| entity_type | value | class |
|---|---|---|
| specialist | cardiologist | Cardiologist |
| procedure | stress test | Cardiothoracic procedures |
| benefit | office visit | Place of service |

*Part 4 covers the matching logic and evaluation.*

---

## Step 5: Resolution and operational metrics

A resolution prompt reads the transcript and returns `RESOLVED`, `NOT_RESOLVED`, or `UNCLEAR`, plus a one-line reason so the label can be audited.

In parallel, operational metrics are computed per call:

| Metric | Definition |
|---|---|
| Contact volume | Count of distinct `call_id` |
| Avg. handle time | Talk + hold + after-call work |
| Avg. hold time | Time the caller is on hold |
| Transfer rate | % of calls transferred |
| Avg. sentiment score | Model-scored sentiment, 0–1 |
| Resolution % | % of calls labeled `RESOLVED` |

---

## Step 6: The consolidation pipeline

At this point there are five tables (summaries, topics, entities, resolution, metrics), and each job finishes at a different time. If the dashboard joined them at query time, it would be **slow, expensive, and could show incomplete data** while a job was still running.

So a final **consolidation pipeline** builds one table for the dashboard.

![Consolidation](/assets/img/call-intelligence/02_consolidation.png)
*Figure 3: The consolidation pipeline runs only after every upstream task succeeds for the run date.*

### Orchestration

```python
# Airflow (simplified)
summary_task >> [topic_task, entity_task]

[topic_task, entity_task, resolution_task, metrics_task] >> consolidate_task
```

With the default `all_success` trigger rule, `consolidate_task` runs only after **every** upstream task has succeeded for that date.

### Table design

```sql
-- Mock schema
CREATE TABLE contact_intel.call_intelligence_dashboard (
  call_date          DATE,
  lob                STRING,
  call_id            STRING,
  member_id          STRING,      -- tokenized / de-identified
  channel            STRING,

  -- AI-generated fields
  concise_summary    STRING,      -- access-controlled
  topics             ARRAY<STRUCT<topic_group STRING, topic STRING>>,
  entities           ARRAY<STRUCT<entity_type STRING, class STRING, value STRING>>,
  resolution         STRING,
  resolution_reason  STRING,
  repeated_issue     INT64,
  transfers          INT64,

  -- Operational metrics
  handle_time_sec    INT64,
  hold_time_sec      INT64,
  transferred        BOOL,
  sentiment_score    FLOAT64,

  load_ts            TIMESTAMP
)
PARTITION BY call_date
CLUSTER BY lob, call_id, member_id
OPTIONS (require_partition_filter = TRUE);
```

| Choice | Reason |
|---|---|
| **Partition by `call_date`** | Every dashboard view filters on dates, so BigQuery scans only the partitions it needs |
| **Cluster by `lob` first** | LOB is the next most common filter |
| **Cluster by `call_id`, `member_id`** | Fast lookups for single-call review and member drill-downs |
| **`require_partition_filter`** | Prevents accidental full-table scans |
| **Topics and entities as `ARRAY<STRUCT>`** | Many per call, while keeping one row per call |

> **Note:** BigQuery supports only one partitioning column, so LOB is handled through clustering rather than as a second partition.

### Loading with an idempotent MERGE

```sql
DECLARE run_date DATE DEFAULT @run_date;

MERGE contact_intel.call_intelligence_dashboard T
USING (
  SELECT
    m.call_date, m.lob, m.call_id, m.member_id, m.channel,
    s.concise_summary, s.repeated_issue, s.transfers,
    tp.topics, e.entities,
    r.resolution, r.resolution_reason,
    m.handle_time_sec, m.hold_time_sec, m.transferred, m.sentiment_score,
    CURRENT_TIMESTAMP() AS load_ts
  FROM contact_intel.call_metrics m
  LEFT JOIN contact_intel.call_summary_parsed s USING (call_id)
  LEFT JOIN contact_intel.call_resolution     r USING (call_id)
  LEFT JOIN (
    SELECT o.call_id,
           ARRAY_AGG(STRUCT(h.topic_group, t AS topic)) AS topics
    FROM contact_intel.call_topic o, UNNEST(o.call_topics) AS t
    JOIN contact_intel.call_topic_hierarchy h ON h.topic = t
    WHERE o.call_date = run_date
    GROUP BY o.call_id
  ) tp USING (call_id)
  LEFT JOIN (
    SELECT call_id,
           ARRAY_AGG(DISTINCT STRUCT(entity_type, class, value)) AS entities
    FROM contact_intel.call_entity
    WHERE call_date = run_date
    GROUP BY call_id
  ) e USING (call_id)
  WHERE m.call_date = run_date
) S
ON  T.call_date = run_date
AND T.call_id   = S.call_id
WHEN MATCHED THEN UPDATE SET
  concise_summary = S.concise_summary, topics = S.topics,
  entities = S.entities, resolution = S.resolution,
  resolution_reason = S.resolution_reason,
  repeated_issue = S.repeated_issue, transfers = S.transfers,
  handle_time_sec = S.handle_time_sec, hold_time_sec = S.hold_time_sec,
  transferred = S.transferred, sentiment_score = S.sentiment_score,
  load_ts = S.load_ts
WHEN NOT MATCHED THEN INSERT ROW;
```

`call_summary_parsed` is a view that decrypts and parses the summary JSON for authorized jobs only. The `T.call_date = run_date` condition keeps the MERGE to one partition, and reruns overwrite instead of duplicating.

---

## Step 7: Querying one table, no joins

**Topic-level KPIs** (multi-label, so count distinct calls):

```sql
SELECT
  t.topic_group,
  t.topic,
  COUNT(DISTINCT call_id)                  AS contact_volume,
  AVG(handle_time_sec)                     AS avg_handle_sec,
  AVG(CAST(transferred AS INT64))          AS transfer_rate,
  AVG(sentiment_score)                     AS avg_sentiment,
  AVG(IF(resolution = 'RESOLVED', 1, 0))   AS resolution_pct
FROM contact_intel.call_intelligence_dashboard,
     UNNEST(topics) AS t
WHERE call_date BETWEEN @start_date AND @end_date
  AND lob = @lob
GROUP BY t.topic_group, t.topic
ORDER BY contact_volume DESC;
```

**Top entity classes:**

```sql
SELECT ent.entity_type, ent.class, COUNT(DISTINCT call_id) AS calls
FROM contact_intel.call_intelligence_dashboard,
     UNNEST(entities) AS ent
WHERE call_date BETWEEN @start_date AND @end_date
  AND lob = @lob
GROUP BY ent.entity_type, ent.class
ORDER BY calls DESC;
```

With one table you can answer questions like *which topics have high volume, long handle time, **and** low resolution?* Those are the best candidates for self-service, agent training, or process fixes.

---

## Step 8: The dashboard

Every view reads from `call_intelligence_dashboard` and shares the same filters (date range, LOB, channel, topic, sentiment), so the numbers always agree.

| View | What it shows |
|---|---|
| **Performance** | KPI cards plus volume over time (DoD / WoW / MoM / QoQ / YoY) |
| **Topic Insights** | Calls by topic group, drill-down to topics, Unfiltered shown separately |
| **Named Entities** | Top classes per entity type, drill-down to raw values |
| **Trends** | Topic volume over time |
| **Contact Table** | Call-level records with AI fields |
| **Session Review** | One call's transcript, summary, topics, entities, and resolution together |

![Performance](/assets/img/call-intelligence/mock_performance.png)
*Figure 4: Performance view (mock data).*

![Topic Insights](/assets/img/call-intelligence/mock_topic_insights.png)
*Figure 5: Topic Insights. Shares can exceed 100% because a call can have several topics (mock data).*

![Named Entities](/assets/img/call-intelligence/mock_named_entities.png)
*Figure 6: Named Entities, with drill-down from a class to the values behind it (mock data).*

![Trends](/assets/img/call-intelligence/mock_trends.png)
*Figure 7: Topic group trends month over month (mock data).*

---

## Lessons learned

1. **Summarize first.** One clean, structured summary makes every downstream model cheaper and more accurate.
2. **Use the simplest tool for each layer.** An LLM where language is messy, a fine-tuned classifier for high-volume topics, a dictionary for fixed categories.
3. **Constrain every output.** Response schemas, fixed taxonomies, and mapping tables. Free text can't be aggregated.
4. **Consolidate before you visualize.** Build one partitioned, clustered table after all jobs finish. Queries are fast and never show half-loaded data.
5. **Design for multi-label.** Calls cover several topics and many entities; count distinct calls.
6. **Keep `call_id` everywhere.** Every output must join back to its source call.
7. **Protect sensitive data.** Encrypt summaries, de-identify member IDs, and restrict access to transcripts and health entities.
8. **Make results auditable.** A single-session view lets users check AI output against the original transcript.

---

## What's next in this series

- **Part 2:** [Structured Call Summaries: Turning Long, Noisy Transcripts into Clean JSON](/posts/2026/10/28/call-intelligence-2-summaries/)
- **Part 3:** [LLM as Teacher, XLNet as Student: Classifying Why Members Call](/posts/2026/10/28/call-intelligence-3-topics/)
- **Part 4:** [Named Entities from Call Summaries: Why We Chose a Dictionary over an LLM](/posts/2026/10/28/call-intelligence-4-entities/)
