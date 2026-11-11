---
layout: post
title: "Is Our Prompt Fair? Checking Call360 Summaries for Bias with LangFair"
summary: "Call360's summaries are written by an LLM from a member's record. Here's how we use LangFair to check that our prompt doesn't add stereotypes, harmful language or invented details, in plain terms."
project: call360
date: 2026-11-11 09:00:00
---

*Part 3 of 3 on Call360. [Part 1](/posts/2026/10/30/call360/) covers the data and API. This part covers how we check the summary prompt for bias.*

> All member details, prompts, and scores in this article are **mock data** created for illustration.

---

## A summary can say more than the record does

Call360 turns a member's history into a short summary, and the summary is what the rep reads before saying hello. Our first prompt wrote that summary from the record alone. It didn't know the member's gender, age, or background, and it wasn't supposed to guess.

Language models are good at filling gaps. Without a check, a summary can quietly add a detail that was never in the record, or describe a group of people in a way that isn't fair. We needed a way to test for that, not just read a few outputs and hope.

---

## What "fair" means here

Take two members with the same facts: the same denied prescription, the same approved authorization, the same open grievance. A fair prompt writes the same summary for both. It doesn't add tone, assumptions, or details that aren't in the record.

That's the bar. We're not asking whether one summary is perfect. We're asking whether the prompt treats records that look alike in the same way.

---

## The tool: LangFair

[LangFair](https://github.com/cvs-health/langfair) is an open-source Python library for checking bias and fairness in LLM applications. It was developed by our governance team, not by the Call360 team. We use it to assess fairness and bias in our prompts, and we didn't write it.

Its main idea is **use-case level** testing: you bring your own prompts, rather than relying on a generic benchmark that may not match what your app actually does.

That fits Call360. The risks in a summary of a member's record aren't the same as the risks in a chatbot, so we test with our own prompts.

Running it looks like this (simplified from the library's summarization demo):

```python
from langfair.auto import AutoEval

ae = AutoEval(
    prompts=prompts,               # one prompt per mock member record
    langchain_llm=llm,             # our model, wrapped as a LangChain chat model
    suppressed_exceptions=suppressed,
)
results = await ae.evaluate(return_data=True)
ae.print_results()
```

Behind those few lines, LangFair generates a summary for each prompt, runs the checks, and returns one set of results.

---

## Step 1: Does the prompt use demographic information?

LangFair starts with a check called **fairness through unawareness**. If the prompt never receives protected attributes such as gender, race, or age, the model can't treat groups differently because of them.

Our prompt passes this check. The member record contains claims, pharmacy, authorization, grievance, and call history, but no demographic fields. For summarization, LangFair then uses toxicity metrics as the main assessment. Counterfactual tests, which compare outputs for records that differ only in a demographic term, are skipped, since there is nothing demographic to swap.

That isn't the end of the story. A name or pronoun can carry demographic signal even when no field does. The stereotype checks look at the summaries themselves, so they catch that too.

<figure class="diagram">
{% include diagrams/fair-decision.svg %}
<figcaption>Figure 1: LangFair's guidance for summarization. Our prompt passes the unawareness check, so the main assessment is toxicity, with stereotype checks added (mock flow).</figcaption>
</figure>

<figure class="diagram">
{% include diagrams/fair-flow.svg %}
<figcaption>Figure 2: How AutoEval runs. Toxicity and stereotype metrics always run. The protected-group check decides whether counterfactual metrics are added or skipped (mock data).</figcaption>
</figure>

---

## Step 2: Is anything harmful?

The **toxicity** checks ask whether a summary contains rude, offensive, or harmful language. LangFair uses a pretrained classifier to score each summary, then reports how often a summary crosses a threshold.

For summarization without demographic inputs, toxicity is LangFair's main assessment. A summary that is accurate but mocking in tone would still be a problem for the rep and the member, so it's the check we rely on most.

---

## Step 3: Does it link groups to stereotypes?

The **stereotype** checks look for a specific failure: a summary linking a group of people to a stereotyped word or idea. LangFair scores each summary and measures whether group words cluster with stereotyped words more for some groups than others.

This is an extra check for us, not the main one. Our record contains no group information, so a summary that introduces a gendered pronoun, an age label, or a group word that isn't in the record is worth flagging. AutoEval runs these checks alongside toxicity either way.

---

## Reading the results

Each metric is a score, and each one has a target. Most of ours target **0**, meaning no flagged outputs.

<figure class="diagram">
{% include diagrams/fair-scoreboard.svg %}
<figcaption>Figure 3: One run of the main checks, with the target for each and the mock result (mock data).</figcaption>
</figure>

In plain words:

- **Toxic fraction:** what share of summaries were flagged as harmful. Target 0.
- **Expected maximum toxicity:** for each prompt, LangFair generates 25 summaries and takes the worst. This score averages those worst cases. Lower is better.
- **Toxicity probability:** the chance that at least one of the 25 summaries for a prompt is harmful. Target 0.
- **Stereotype association:** the extra check. It measures how strongly group words appear alongside stereotyped words. Lower is better.

A score isn't a verdict on its own. When a number is above target, LangFair can list the exact summaries behind it, so a person can read them and decide what went wrong.

---

## What we changed

The risk these checks look for is not in what we pass in. It's in what the model adds. That's why the pass 1 prompt has a guardrail: use only the context, never add facts, and don't describe the member beyond what the record says. The Walkthrough section of Part 1 shows that guardrail in the full prompt.

A guardrail is a claim about the prompt. The checks are how we test that the claim holds, and they're what we'd rerun after any change to the wording.

---

## Keeping it in the pipeline

A fairness check is only useful if it runs again when something changes. The approach we're building toward runs LangFair:

- **On every prompt change.** A new prompt version should pass before it reaches the rep.
- **On a sample of production summaries.** That would catch drift when the data changes.
- **Before each release.** Results are exported to a file a reviewer can read.

LangFair also keeps the exact summaries behind each score, so a reviewer can start from a flagged row, not from a number.

---

## Limits

LangFair gives us measurements, not a certificate. The checks depend on the classifiers and word lists behind them, and they can miss things they weren't built to see. We treat the scores as a starting point for human review.

For the same reason, we don't report a single "fair" or "not fair" label. We report the scores, what they measure, and what we did about any that were off target.

---

## Lessons learned

1. **Test your own prompt, not a generic one.** The risks in a member summary are specific to that task.
2. **A passing check can still hide a gap.** The unawareness check passing doesn't mean names and pronouns are harmless. The stereotype checks catch what the first check can't.
3. **Make the rule visible in the prompt.** A guardrail the model can read is easier to test than a policy written somewhere else.
4. **Keep the summaries, not just the scores.** A number tells you something is off. The text tells you what to fix.

---

## Conclusion

Call360's summaries are only useful if the rep can trust them. LangFair gives us a repeatable way to check that our prompt doesn't add what the record doesn't say, and to show the evidence when we change it.

*LangFair was developed by our governance team; this article describes how the Call360 team uses it. All member details, prompts, and scores in this article are fictional and for illustration only. Views are my own.*
