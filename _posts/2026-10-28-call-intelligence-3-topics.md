---
layout: post
title: "LLM as Teacher, XLNet as Student: Classifying Why Members Call"
summary: "Using an LLM to label training data and a fine-tuned XLNet model to tag every call with the topics discussed."
date: 2026-10-28 09:02:00
---

*Part 3 of a 4-part series on conversation intelligence. Builds on the structured summaries from Part 2.*

> All examples, topic names, numbers, and screens in this article are **mock data** created for illustration.

---

## The question behind every contact center

To run a contact center well, you first have to answer one question: **why are people calling?**

Most contact centers rely on **manual disposition codes** that agents select after each call. They're slow to capture, inconsistent between agents, and usually too broad for real analysis. "Claims" tells you very little. "Asking why a claim was denied" tells you where to look.

The call topic model tags **every call** with topics from a **pre-defined taxonomy**, so topics can be tracked, trended, and acted on. That makes it the foundation for finding call-deflection opportunities and widespread pain points.

One subtlety worth stating up front: the model tags topics **discussed in the call**, whether the caller or the agent raised them. It isn't limited to the caller's original reason for calling.

---

## The design: an LLM labels, a smaller model learns

You might expect us to send every summary to an LLM and ask for its topics. We use the LLM differently.

![Teacher-student](/assets/img/call-intelligence/p3_teacher_student.png)
*Figure 1: The LLM labels a training set; a fine-tuned XLNet model classifies calls in production.*

The pipeline has two stages:

1. **Teacher:** an LLM labels a set of 10,000 structured summaries with topics, using carefully designed few-shot prompts.
2. **Student:** a pre-trained **XLNet** transformer is fine-tuned on those labels as a **multi-label classifier**. XLNet is what runs every day in production.

This is a common pattern for high-volume classification. The expensive, flexible model is used once to create labeled data. A smaller, specialized model then does the repetitive work, with the same input always giving the same output.

---

## Input: the structured summary, not the transcript

The model reads two sections of the structured summary from Part 2:

- the **concise summary**
- the **call intents**

**Example input (mock):**

> *Concise summary:* The caller is trying to book an appointment with a dermatologist at a local clinic. Her first-choice doctor was fully booked, so she booked with a new doctor whose details may not be in the directory yet. The agent confirms the first doctor is in-network and explains that the new doctor's information needs to be updated.
>
> *Intents:* The caller wants to know if a dermatologist at the clinic is in-network. The caller wants to know if the clinic is part of the plan's network.

Because Part 2's summary prompt asks for granular, keyword-rich intents, this input is already clean and focused. That is a big reason the topic model works.

---

## Stage 1: Labeling training data with an LLM

### Split the summary into components

Each summary is split into **components**: one per intent, plus the concise summary. Each component is labeled separately.

### Two-step chain-of-thought: group first, then topic

The taxonomy is large. Asking an LLM to choose from every topic at once invites confusion between similar topics. So each component goes through **two API calls**:

![Two-step labeling](/assets/img/call-intelligence/p3_two_step_labeling.png)
*Figure 2: Predict broad topic groups first, then choose specific topics only from those groups.*

1. **Predict topic groups** from a fixed list of high-level groups.
2. **Predict topics** from only the candidate topics mapped to those groups. The group-to-topic mapping is curated by hand.

Narrowing the choices first means the second call picks from a short, relevant list instead of the full taxonomy.

Each call has a cap on how many labels it can return:

| API call | Max predictions |
|---|---|
| Intent → group | 3 |
| Intent → topic | **1** |
| Concise summary → group | 3 |
| Concise summary → topic | 3 |

Two rules shape the output:

- **One topic per intent.** Each intent is a single question, so it gets exactly one topic.
- **The summary adds, it doesn't repeat.** When labeling the concise summary, topics already found from the intents are excluded. That way the summary pass picks up only *additional* topics, such as something the agent raised.

### Few-shot examples chosen per input

The prompt includes hand-curated examples. Rather than using the same examples every time, the most relevant ones are selected for each input using **n-gram overlap scoring**. Each prompt gets:

- **at least 3 examples** covering **at least 2 different topics**, and
- **at least 1 example with no label**, which teaches the model that "None" is a valid answer for noisy or off-taxonomy text.

### The prompt (simplified)

```text
You are a data labeler categorizing a caller's intent: any action the
caller takes or intends to take during a call with a health insurance
call center.

YOUR KNOWLEDGE IS STRICTLY LIMITED TO THE CALL INTENT AND THE EXAMPLES
PROVIDED. DO NOT MAKE ASSUMPTIONS BEYOND WHAT IS EXPLICITLY STATED.

From the candidate list, choose the topic that best describes the
caller's intent. If the intent does not match any candidate (too niche,
ambiguous, or undefined), respond "None".

Candidate topics and definitions:
{candidate_definitions}

Examples:
{best_few_shot_examples}

Intent: {intent}
Topic:
```

### API settings

| Setting | Value |
|---|---|
| Model | Gemini-family LLM |
| Temperature | 0.1 |
| Top-p | 0.95 |
| Retries | Exponential backoff |

All predicted topics for a call are then combined into one label list, which becomes a row of training data: the summary text, plus a **0/1 flag for every topic**.

### Checking the labels before training

A student model can only be as good as its teacher's labels. Before training, each label in the training data is checked: **an LLM verifies whether the predicted topic is actually present in the summary.** This produces a precision score for each topic, so weak topics are caught before any training happens.

---

## Stage 2: Fine-tuning XLNet

### Small and large topics

Topic volumes are very uneven. A handful of topics cover a large share of calls, while many others are rare. If you train one model on all of them, the frequent topics dominate and rare ones get lost.

So topics are split by volume and trained separately:

![Small vs large topics](/assets/img/call-intelligence/p3_small_large.png)
*Figure 3: Rare and common topics are modeled separately so rare topics don't get drowned out.*

| Group | Definition | Training target |
|---|---|---|
| **Small topics** | Less than 1% of total calls | 0/1 vector over small topics only |
| **Large topics** | 1% of calls or more | 0/1 vector over large topics only |

In both cases the input is the same: the concise summary plus the call intents.

### Multi-label classification

Each call can have several topics, so the model is **multi-label**: it scores every topic on its own instead of choosing a single winner. Here is a minimal sketch of the pattern using Hugging Face Transformers. It's illustrative only, not our production code:

```python
from transformers import XLNetTokenizer, XLNetForSequenceClassification

tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
model = XLNetForSequenceClassification.from_pretrained(
    "xlnet-base-cased",
    num_labels=len(TOPICS),                       # one output per topic
    problem_type="multi_label_classification",    # sigmoid per label
)

# training: text = concise summary + intents, labels = 0/1 float vector
inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
outputs = model(**inputs, labels=label_vectors)   # BCE-with-logits loss

# inference: every topic whose probability clears a threshold
probs = outputs.logits.sigmoid()
predicted = [[TOPICS[j] for j, p in enumerate(row) if p > THRESHOLD]
             for row in probs]
```

---

## Running in production

The fine-tuned model runs as a **daily batch** on new structured summaries. Because transcripts and summaries need to be complete first, topics land **a few days after the call**.

Output, one row per call:

| Field | Description |
|---|---|
| `call_id` | Unique identifier for the call |
| `call_date` | Date the call occurred |
| `call_topics` | List of topics predicted for the call |
| `run_time` | When the model ran for this call |

A separate **topic hierarchy table** maps each topic to its group. Keeping the hierarchy in its own table means topics can be regrouped or renamed for reporting without re-running the model.

```sql
-- Count distinct calls per topic and group
SELECT
  h.topic_group,
  t AS topic,
  COUNT(DISTINCT o.call_id) AS calls
FROM contact_intel.call_topic o,
     UNNEST(o.call_topics) AS t
JOIN contact_intel.call_topic_hierarchy h
  ON h.topic = t
WHERE o.call_date BETWEEN @start AND @end
GROUP BY h.topic_group, topic
ORDER BY calls DESC;
```

> **Counting tip:** because calls are multi-label, topic shares can add up to more than 100%. Count `DISTINCT call_id` per topic, and say so on the dashboard.

### "Unfiltered" calls

Some calls don't map to any topic. These are reported as **Unfiltered**, rather than forced into the closest category. They come from two sources:

- **Calls without content**, such as calls disconnected within seconds.
- **Calls about topics not in the taxonomy yet**.

The first group is noise. The second is a signal: if Unfiltered volume grows, the taxonomy is missing something.

---

## How we measure quality

![Evaluation](/assets/img/call-intelligence/p3_evaluation.png)
*Figure 4: Two-step evaluation: the training data first, then the model.*

### The metrics, in plain language

- **Precision: how often are we right?** Put every call the model labeled "asking why a claim was denied" in a bucket. Precision is how *pure* that bucket is.
- **Recall: how many did we find?** Picture the whole river of calls. Recall is how many of the true "denied claim" calls the net actually caught.
- **F1: the balance.** It's hard to maximize both at once. F1 measures how well you fill the bucket with only the right calls *and* catch most of them.

### Step 1: Training data (machine evaluation)

For each label the LLM produced, a second LLM check confirms whether the topic is really in the summary. Precision is calculated per topic and then averaged across topics **without weighting**, so rare topics count as much as common ones.

### Step 2: Model (human validation)

1. Sample about **200 calls**, chosen so **every topic appears at least 10 times**.
2. For each predicted topic, a reviewer marks whether it's actually present in the **transcript** (precision).
3. The reviewer also notes topics in the transcript that the model **missed** (recall).
4. Compute precision, recall, and F1 per topic, then average across topics, again unweighted.

Two choices are worth copying:

- **Stratified sampling** ("each topic at least 10 times") gives every topic enough examples to be measured. A random sample would be mostly the top few topics.
- **Validating against the transcript, not the summary**, measures the whole chain end to end, including anything the summary missed.

### Keeping it current

The model is re-evaluated **every six months**. Each cycle:

- improves accuracy for topics that are underperforming, and
- adds new topics based on user requests and emerging needs, often surfaced by the Unfiltered bucket.

---

## Putting topics to work

Joined with operational metrics in the consolidated table (Part 1), topics show not just *what* members call about but *what each topic costs*:

![Topic drilldown](/assets/img/call-intelligence/mock_topic_drilldown.png)
*Figure 5: Drill-down from a topic group to specific topics, with operational metrics (mock data).*

- **High volume and simple** → self-service candidates.
- **High transfer rate or low resolution** → broken handoffs or unclear processes.
- **Rising trend** → an emerging issue to investigate early.
- **After a fix ships** → watch that topic's volume to measure impact.

---

## Lessons learned

1. **Use the LLM as a teacher, not necessarily as the production model.** Let it create labels, then train a smaller model to apply them every day.
2. **Narrow before you choose.** Picking a topic group first, then a topic, beats choosing from the full taxonomy in one step.
3. **Pick few-shot examples per input**, and include a "no label" example so "None" stays a real option.
4. **Check the teacher's labels.** Errors in training data become errors in the model.
5. **Model rare and common topics separately.** Otherwise rare topics disappear.
6. **Sample so every topic is measured**, and average without weighting.
7. **Treat Unfiltered as feedback.** It shows where the taxonomy needs to grow.

---

**Next: Part 4: Named Entities from Call Summaries: Why We Chose a Dictionary over an LLM**
