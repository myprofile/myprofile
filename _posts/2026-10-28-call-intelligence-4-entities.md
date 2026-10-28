---
layout: post
title: "Named Entities from Call Summaries: Why We Chose a Dictionary over an LLM"
summary: "Why matching named entities from call summaries against a curated dictionary beat using an LLM for this job."
project: call-intelligence
date: 2026-10-28 09:03:00
---

*Part 4 of a 4-part series on conversation intelligence. Builds on the structured summaries from Part 2.*

> All examples, keywords, numbers, and screens in this article are **mock data** created for illustration.

---

## From "why" to "about what"

Knowing a member called about the *status of a denied claim* is useful. Knowing the denied claim was for a **cardiologist** visit is actionable.

Topics (Part 3) tell you *why* someone called. Named entities add the next level of detail: the specific **benefits, specialists, procedures, and diseases** mentioned in the conversation.

![Value to class](/assets/img/call-intelligence/p4_value_class.png)
*Figure 1: One sentence can produce entities of all four types. Each mention (value) is mapped to a standard category (class).*

Notice that "office visit" appears twice. As a **benefit** it maps to a place-of-service category; as a **procedure** it maps to *physician visit*. The same words mean different things depending on the entity type, and the design has to allow for that.

---

## The surprising design choice: no LLM

The summary model is an LLM. The topic model is trained on LLM-generated labels. The entity model uses no LLM at all.

It's a **dictionary-based extractor**: curated keyword tables, exact matching, and a longest-match rule. Here's why that was the right call:

| Concern | LLM extraction | Dictionary extraction |
|---|---|---|
| **Output consistency** | Same concept can come back as "PT", "physio", "physical therapy" | Every keyword maps to one fixed class |
| **Fixed categories** | Must be enforced and validated | Built in: the class *is* the mapping |
| **Explainability** | Hard to say why something was extracted | Every match traces to one row in a table |
| **Cost** | A second LLM call per call | Near zero once summaries exist |
| **Fixing errors** | Rewrite and re-test the prompt | Add or edit a row in the table |
| **Recall on new phrasing** | Strong | Only as good as the keyword list |

The trade-off is clear: you give up some recall on unusual wording and gain **consistent, explainable, cheap** output. For counting entities across millions of calls on a dashboard, consistency matters most.

The heavy lifting is already done upstream. The LLM summary (Part 2) cleans up transcription errors and writes in plain, consistent language, which makes simple matching work much better than it would on raw transcripts.

---

## How it works

![Entity pipeline](/assets/img/call-intelligence/p4_entity_pipeline.png)
*Figure 2: The entity pipeline runs daily on new structured summaries.*

### 1. Mapping tables: the real model

There's one keyword table per entity type. Each row maps a keyword to a class:

| kw | kw_lemmatized | category |
|---|---|---|
| chemo treatment | chemo treatment | Infusion therapy |
| oxygen concentrator | oxygen concentrator | DME |
| home oxygen | home oxygen | DME |
| hearing test | hearing test | Hearing |
| test for hearing | test hearing | Hearing |

*Illustrative extract of a benefit mapping table.*

The **lemmatized** column normalizes word forms, so "test for hearing" and "hearing tests" can land on the same class. These tables are where the domain knowledge lives. Improving the model mostly means improving them.

### 2. Expand keyword variants

Before matching, each keyword and its lemmatized form are expanded into **singular and plural variants**. That way "panel" also matches "panels", without anyone having to list both forms by hand.

### 3. Exact match, longest span first

For each summary:

1. Find every exact match from the expanded keyword set.
2. **Rank matches by length, longest first.**
3. Assign matches one at a time. Once a span of text is claimed, shorter keywords inside it are skipped.
4. Attach the class from the mapping table.

![Longest match](/assets/img/call-intelligence/p4_longest_match.png)
*Figure 3: Longest-span-first matching keeps "obstetric panels" from also being counted as a generic lab "panel".*

Without step 2, "two obstetric panels" would produce **two** entities: a maternity procedure and a generic laboratory study. That's wrong, and it inflates counts. Longest-match-first is a small rule with a big effect on precision.

```python
import re

def expand(mapping: dict[str, str]) -> dict[str, str]:
    """Add simple singular/plural variants for each keyword."""
    out = {}
    for kw, cls in mapping.items():
        out[kw] = cls
        if kw.endswith("s"):
            out[kw[:-1]] = cls        # plural -> singular
        else:
            out[kw + "s"] = cls       # singular -> plural
    return out

def extract(text: str, mapping: dict[str, str]) -> list[tuple[str, str]]:
    text_l = text.lower()
    taken = [False] * len(text_l)
    results = []
    # longest keywords first
    for kw in sorted(mapping, key=len, reverse=True):
        for m in re.finditer(rf"\b{re.escape(kw)}\b", text_l):
            s, e = m.span()
            if any(taken[s:e]):
                continue                      # span already claimed
            taken[s:e] = [True] * (e - s)
            results.append((text[s:e], mapping[kw]))
    return results

procedures = expand({"panel": "Laboratory studies",
                     "obstetric panel": "Maternity procedures"})
extract("The member went to get two obstetric panels", procedures)
# -> [('obstetric panels', 'Maternity procedures')]
```

*A simplified sketch. Production versions also handle lemmatization and irregular plurals.*

The same extractor runs once per entity type, each with its own mapping table. That's how "office visit" can be both a benefit and a procedure.

---

## Output tables

Each entity type writes to its own table, and a combined table stacks all four with an `entity_type` column:

| Column | Type | Description |
|---|---|---|
| `call_id` | STRING | Unique identifier for the call |
| `call_date` | DATE | Date the call occurred |
| `entity_type` | STRING | benefit, specialist, procedure, or disease (combined table only) |
| `class` | STRING | Standard category from the mapping table |
| `value` | STRING | The text as it appeared in the summary |
| `run_date` | DATE | Date the model ran |

Keeping both **value** and **class** is deliberate:

- **class** powers dashboards: a short, stable list you can count and trend.
- **value** powers drill-downs and debugging: what people actually said.

```sql
CREATE TABLE contact_intel.call_entity (
  call_id     STRING,
  call_date   DATE,
  entity_type STRING,   -- benefit | specialist | procedure | disease
  class       STRING,
  value       STRING,
  run_date    DATE
)
PARTITION BY call_date
CLUSTER BY entity_type, class, call_id;
```

---

## Querying entities

**All entities for one call:**

```sql
SELECT entity_type, class, value
FROM contact_intel.call_entity
WHERE call_date = @call_date
  AND call_id   = @call_id
ORDER BY entity_type, class;
```

**All calls that mention a class** (e.g., any dental benefit):

```sql
SELECT DISTINCT call_id
FROM contact_intel.call_entity
WHERE call_date BETWEEN @start AND @end
  AND entity_type = 'benefit'
  AND class = 'Dental';
```

**All calls that mention a specific value** (e.g., "sealants"):

```sql
SELECT DISTINCT call_id
FROM contact_intel.call_entity
WHERE call_date BETWEEN @start AND @end
  AND entity_type = 'benefit'
  AND value = 'sealants';
```

**Top classes, counting each call once:**

```sql
SELECT class, COUNT(DISTINCT call_id) AS calls
FROM contact_intel.call_entity
WHERE call_date BETWEEN @start AND @end
  AND entity_type = 'procedure'
GROUP BY class
ORDER BY calls DESC;
```

The same keyword can match several times in one summary (the concise summary, then again in each intent). Use `COUNT(DISTINCT call_id)` for "how many calls," not `COUNT(*)`.

![Named Entities dashboard](/assets/img/call-intelligence/mock_named_entities.png)
*Figure 4: Dashboard view of procedure classes, with drill-down to the raw values behind a class (mock data).*

---

## How we measure quality

Entities are evaluated against a hand-labeled set of a few hundred summaries.

![Evaluation](/assets/img/call-intelligence/p4_evaluation.png)
*Figure 5: Evaluation flow.*

1. **Label** every (value, class) pair that should be extracted, for all four entity types.
2. **Compare** with the model's output to count true positives, false positives, and false negatives.
3. **Compute** precision, recall, and F1.

| Metric | Question |
|---|---|
| **Precision** | Of the (value, class) pairs extracted, how many are correct? |
| **Recall** | Of the pairs that should have been found, how many were? |
| **F1** | Balance of the two |

We report each metric two ways, because they answer different questions:

- **Micro average:** pool every entity together. This tells you how well the system performs **overall**, and it's dominated by the most common classes.
- **Macro average:** compute the metric for each class, then average them equally. This shows whether the system **struggles with specific classes**, such as rare procedures that are drowned out in the micro number.

When macro recall is noticeably lower than micro recall, the cause is usually rare classes with thin keyword coverage. The fix is to add keywords, not to change the algorithm.

A pattern that showed up in our evaluation: diseases were the hardest entity type. Callers describe symptoms ("chest pain") more often than diagnoses, and disease vocabulary varies a lot. Specialists were the easiest: there are only so many ways to say "cardiologist."

### What errors look like

Reviewing individual misses is where the improvements come from:

- **False negative:** a misspelling the summary model passed through (for example, a garbled form of "echocardiogram") doesn't match any keyword. *Fix: add the common variant, or tighten the summary prompt's "no transcription errors" rule.*
- **False positive:** a generic word like "disease" matches a broad catch-all class. *Fix: remove or restrict overly generic keywords.*

Both fixes are a table edit, with no retraining or prompt rewrite.

---

## Where entities pay off

Entities are most useful **combined with topics**:

- Which **procedures** drive the most prior-authorization calls? Those are candidates for clearer pre-service guidance.
- Which **specialist** types come up most in "find a provider" calls? These point to network or search-tool gaps.
- Which **benefits** cause the most coverage questions? These point to plan documents that need clearer wording.

---

## Privacy note

Disease and procedure entities are sensitive health information. Treat entity tables with the same controls as transcripts and summaries: restrict access by role, report aggregates rather than individuals, and apply minimum-count thresholds before showing rare values on a dashboard.

---

## Lessons learned

1. **Not every step needs an LLM.** Use the LLM where language is messy (summaries), a trained classifier for high-volume topics, and deterministic rules where consistency matters (entities).
2. **The mapping table is the model.** Invest in keyword coverage and clear classes.
3. **Longest match first.** It prevents double counting and protects precision.
4. **Keep value and class.** Class for dashboards, value for drill-down and debugging.
5. **Report micro and macro.** Micro shows overall health; macro exposes weak classes.
6. **Count distinct calls.** Keywords repeat within a summary.
7. **Make fixes cheap.** When an error is a table edit, the model gets better every week.

---

## Series wrap-up

Across the four parts, one pattern repeats: **turn unstructured conversation into constrained, structured data, one layer at a time, and pick the simplest tool that works for each layer.**

1. **Transcripts → structured summaries:** an LLM, because conversations are messy.
2. **Summaries → topics:** an LLM labels training data, and a fine-tuned XLNet model classifies every call.
3. **Summaries → entities:** a dictionary, because consistency beats flexibility here.
4. **Everything → one consolidated, partitioned table:** fast, consistent dashboards.
