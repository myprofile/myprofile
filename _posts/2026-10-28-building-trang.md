---
layout: post
title: "Building Trang: A Generative Agent That Bakes, Remembers and Talks"
summary: "How I gave an AI persona a past, and what it took for her to answer like she'd lived it."
date: 2026-10-28 11:00:00
code:
  - { label: "Code: Interview agent", url: "https://github.com/Sim-Intelligence-Lab/sim_interview_agent" }
  - { label: "Code: Baking agent", url: "https://github.com/Sim-Intelligence-Lab/sim_baking_agent" }
---

*Part 2 of 2 on generative agents. Part 1: [Generative Agents and Simulation](/posts/2026/10/28/generative-agents-and-simulation/), the ideas behind the code.*

Trang started out as a baker.

She lived in an 8×8 pixel kitchen, a 30-year-old at the bakery on the corner, patient and a little perfectionist. She could follow a cake recipe step by step, cheerfully explaining each move. Then I asked her to bake a pizza, and watched the flour in her bowl climb to 3,325 grams out of 500. Every turn she promised she'd get it right this time. She meant it. She just couldn't.

That's when it clicked. Sounding human isn't enough. To act like a person, an agent needs a world that pushes back, and a memory it can actually use.

So I built Trang again. This time she had a whole life behind her, 91 memories of it, and a way to search them. I sat down to interview her:

> **Interviewer:** What have you worked on in your law school's Environmental Law Clinic?
>
> **Trang:** In my law school's Environmental Law Clinic, I helped draft public comments opposing a permit for a factory accused of polluting a nearby river.

It was the right answer. Not a guess or a made-up story, but one of her memories, found among everything she "knows" and told back in her own voice.

This post follows both versions of Trang, and by the end you'll have built the core pieces yourself:

- a **memory stream** that stores her experiences,
- **retrieval** that finds the right memories for each question,
- **reflection** that turns memories into insights.

The result is a chatbot that talks with you using the memories you give it.

**Part 1** is Trang in the kitchen. **Part 2** is the interview.

Everything runs locally on [Ollama](https://ollama.com). No API key, no cloud bill.

## Part 1: Trang in the kitchen

In the first project, Trang is a 30-year-old baker at a neighborhood bakery: patient, careful, a bit of a perfectionist. She lives in an 8×8 pixel kitchen and bakes a cake or a pizza from a recipe stored in her memory.

<video src="/assets/img/building-trang/trang-baking-pizza.mp4" poster="/assets/img/building-trang/trang-baking-pizza-poster.png" autoplay muted loop playsinline aria-label="Trang starting a pizza: she preheats the oven, walks to each dough ingredient, and the ingredient bars fill green as steps get checked off" style="width:100%;height:auto"></video>

*Trang baking a pizza (2× speed, thinking pauses cut).*

Every turn is the same loop: she perceives the kitchen, retrieves the recipe from memory, decides what to do, and a second model call turns her words into actions that a rulebook checks.

<figure class="diagram">
{% include diagrams/trang-one-turn.svg %}
</figure>

Her "memory" here is simple: a text file of unrelated articles with the recipe buried inside, and retrieval just looks for the recipe's headings. That was enough to bake, but it taught me where small-model agents break:

- **She needed an undo.** When her words were misread ("325 ml of water" became `add 325 salt`), she could only add more. I gave her a "start over" action.
- **The world had to talk back.** Phrases that matched nothing used to do nothing, silently, so she repeated herself forever. Now the kitchen says what didn't work.
- **Rules beat prompts.** She kept recounting finished steps and the kitchen counted them twice. Telling her not to failed. Refusing extra flour in code worked.

The bigger lesson: an agent that acts needs a world that keeps state. An agent that talks needs a real memory. That's Part 2.

## Part 2: Talking to Trang

For the second project I gave Trang a different life, one with enough memories to make conversation interesting. In Interview Studio, she grew up in Huntsville, Alabama, exploring forests with her grandmother. She became an environmental advocate, ran for city council, lost, and went to law school to study environmental law.

She has two kinds of knowledge:

- **A profile** (`scratch.json`): a self-description in her own words, a speech pattern, and a fact sheet.
- **A memory stream**: 91 memories, from childhood weekends in the forest to her Sunday law-school study group.

You interview her with 18 prepared questions, 12 factual and 6 reflective. Under every answer, the app shows the memories she used and how each one scored, plus a "How Trang answered" panel with the full prompt and the model's raw reply.

<video src="/assets/img/building-trang/interviewing-trang.mp4" poster="/assets/img/building-trang/interviewing-trang-poster.png" autoplay muted loop playsinline aria-label="Interviewing Trang: a question is asked, she answers, and the memories she used appear with their scores" style="width:100%;height:auto"></video>

*Interviewing Trang in Interview Studio: she answers, and the memories she used appear with their scores.*

### Setup

```bash
ollama pull qwen3:4b-instruct      # the language model, about 2.5 GB
ollama pull qwen3-embedding:0.6b   # the embedding model, about 640 MB
pip install -r requirements.txt
PORT=5050 python interview_app.py
```

The core lives in two files: `generative_agent/modules/memory_stream.py` (storing, scoring, reflecting) and `generative_agent/modules/interaction.py` (answering). Let's build it up one module at a time.

## Step 1: The memory stream

A generative agent's memory is a list of plain-English records called nodes. Each one carries three things that make retrieval possible later:

- **When** it was formed and last recalled, as time steps.
- **How important** it is, from 0 to 100.
- **An embedding**: a list of numbers that captures its meaning, stored alongside.

```python
class ConceptNode:
    def __init__(self, node_dict):
        self.node_id = node_dict["node_id"]
        self.node_type = node_dict["node_type"]        # "observation" or "reflection"
        self.content = node_dict["content"]
        self.importance = node_dict["importance"]
        self.created = node_dict["created"]
        self.last_retrieved = node_dict["last_retrieved"]
        self.pointer_id = node_dict["pointer_id"]      # evidence, for reflections
```

Remembering something is two lines: rate it, then store it with its embedding.

```python
def remember(self, content, time_step=0):
    score = generate_importance_score([content])[0]
    self._add_node(time_step, "observation", content, score, None)

# inside _add_node:
self.seq_nodes += [new_node]
self.embeddings[content] = get_text_embedding(content)
```

The language model does the rating. The prompt gives it anchors so scores mean something:

```text
Score each numbered observation above by how important it would be to a
person's long-term memory. Routine, easily forgotten events should score
from 0 to 15. Major life events, enduring relationships, or defining
beliefs should score from 70 to 100. Judge importance, not how interesting
the wording is.
```

Here's how Trang's memories came out at both ends:

```text
85  As a child, Trang spent countless weekends with her grandmother, exploring the forests...
85  The sudden passing of her grandmother during her junior year deeply affected Trang...
85  Graduating with honors from the University of Alabama, Trang delivered a commencement speech...
10  Trang spent a late night in the law library outlining her Torts notes before an exam...
10  Trang formed a study group with three law school classmates who meet every Sunday...
```

The big life moments sit at the top and the routine law-school nights at the bottom, which is what we want. Adding a new memory goes through the same path:

```python
trang = GenerativeAgent("interviewees", "trang")
trang.remember("Trang won her law school's moot court final.")
trang.save("interviewees", "trang")
```

## Step 2: Retrieval

Ninety-one memories won't fit in a good prompt, and most are irrelevant to any one question. So for every question, Interview Studio scores all of them and keeps the top five.

<figure class="diagram">
{% include diagrams/trang-interview-studio.svg %}
</figure>

The score blends three signals from [the generative agents paper](https://arxiv.org/abs/2304.03442). Each has its own small function:

```python
def extract_recency(seq_nodes):
    most_recent = max(node.last_retrieved for node in seq_nodes)
    return {node.node_id: 0.99 ** (most_recent - node.last_retrieved)
            for node in seq_nodes}

def extract_importance(seq_nodes):
    return {node.node_id: node.importance for node in seq_nodes}

def extract_relevance(seq_nodes, embeddings, focal_pt):
    focal_embedding = get_text_embedding(focal_pt)
    return {node.node_id: cos_sim(embeddings[node.content], focal_embedding)
            for node in seq_nodes}
```

- **Recency** decays by 0.99 per time step since a memory was last recalled.
- **Importance** is the 0–100 rating from Step 1.
- **Relevance** is the cosine similarity between the memory's embedding and the question's.

Each set of scores is scaled to 0–1 across all memories so they're comparable, then combined:

```text
score = 3 × relevance + 0.5 × importance + 0.5 × recency      (max 4)
```

Relevance gets six times the weight of the others. In an interview, being on topic matters most.

Here's the real ranking for the opening question, "What have you worked on in your law school's Environmental Law Clinic?":

```text
score  relevance  importance  memory
3.38   1.00       0.27 (30)   Trang joined her law school's Environmental Law Clinic...
3.31   0.85       1.00 (85)   Reviewing environmental regulations at her nonprofit...
3.27   0.89       0.73 (65)   Trang was accepted into law school...
3.07   0.86       0.47 (45)   Trang landed a summer internship with a public-interest...
2.94   0.75       0.87 (75)   In high school, Trang joined the Environmental Club...
```

The winning memory is rated just 30 out of 100. Seventy-four of her 91 memories rank higher on importance. It wins because it's exactly on topic. Meanwhile, a high-school memory rated 75 only makes fifth place. That's the blend doing its job: importance breaks ties, and relevance decides.

One detail: recency is 0.5 for every memory here. Each interview question is answered on its own, at the same time step, so all memories look equally fresh. In a long-running agent, recency would start to matter.

## Step 3: Reflection

Memories are specific: a lost election, a law school application, a voter registration drive. People also carry bigger ideas built from many such moments. Reflection is how an agent forms those.

It reuses retrieval. Give it an anchor, a topic to think about, and it:

1. Retrieves the 10 memories most related to the anchor.
2. Asks the model for a few first-person insights that follow from them.
3. Rates each insight and stores it as a new "reflection" node, pointing back to its evidence.

```python
def reflect(self, anchor, reflection_count=5, retrieval_count=10, time_step=0):
    records = self.retrieve([anchor], time_step, retrieval_count)[anchor]
    record_ids = [i.node_id for i in records]
    reflections = generate_reflection(records, anchor, reflection_count)
    scores = generate_importance_score(reflections)
    for count, reflection in enumerate(reflections):
        self._add_node(time_step, "reflection", reflections[count],
                       scores[count], record_ids)
```

The prompt is short: "Above are observations about a fictional human subject. Write a list of reflections (in first person voice, from the perspective of the subject) that you can infer from the observations above about the subject on the following anchoring topic."

I asked Trang to reflect on "What has losing the city council race taught me?" One of her three insights:

> "Losing the city council race didn't break my belief in public service—it actually deepened my commitment to grassroots activism. I saw that policy change doesn't happen through a single vote, but through sustained engagement, listening, and building trust with communities that have long been overlooked."

No memory says that. She drew it from ten, including the loss itself, her decision to apply to law school, and a citywide voter registration drive she led. The model rated these insights 85 to 95, higher than almost all of her raw memories. Saved, they would rank near the top whenever someone asked about politics or setbacks.

Interview Studio has `reflect()` in its code but doesn't call it from the app yet, so every answer comes from raw memories. Wiring it in is the first exercise at the end.

## Step 4: The chatbot

With memory and retrieval in place, answering is a well-built prompt. Interview Studio assembles Trang's description from three sections:

```python
sections = [
    f"Self-description:\n{agent.scratch.self_description}",
    f"Speech pattern:\n{agent.scratch.speech_pattern}",
    "Relevant memories:\n" + "\n".join(f"- {memory}" for memory in memories),
]
```

Those go into the answer prompt, along with the conversation and a set of rules. The rules are where most of the reliability comes from:

```text
Speak in first person, with the described speech pattern. Answer the latest
question directly in one or two sentences. Treat the persona and memories as
the complete factual record. For a factual question, use the most directly
relevant memory as the answer and add no new detail. ... If a requested
detail is absent, say you do not recall it.

Return only a valid JSON object with one string field:
{"utterance": "Your reply here."}
```

The app also tells the model what kind of question it is. Factual questions get "answer using only what the memories explicitly state". Reflective ones get "explain what those documented experiences taught you. Do not invent events."

You can see the effect in the opening answer. The clinic memory says Trang "helped draft public comments opposing a permit for a factory accused of polluting a nearby river." Her answer says the same thing in first person, and nothing more: no invented dates, no extra cases. For a persona that's supposed to be grounded, that restraint is the feature.

Asking for JSON keeps the output clean. A small model sometimes wraps answers in commentary, and a single `utterance` field gives the app one thing to extract and show.

Finally, the **"How Trang answered"** panel makes the whole pipeline visible: the memory search and its scores, the full prompt, the model's raw reply, and how long each step took. When an answer looks wrong, you can see whether retrieval picked the wrong memories or the model ignored the right ones. It turns "the bot said something odd" into a question you can answer.

## What I learned

1. **Relevance decides, importance breaks ties.** The best answer to the clinic question came from a memory rated 30 out of 100. Weight relevance highest.
2. **Retrieval is what the agent thinks about.** The model only knows the five memories you hand it. Get those right and most answers get right.
3. **Reflection adds what memories can't.** One extra model call turned ten scattered memories into a belief about public service that none of them states.
4. **Rules in the prompt shape honesty.** "Treat the memories as the complete factual record" and "say you do not recall" kept Trang from inventing details.
5. **Agents that act need a world that pushes back.** In the kitchen, an undo, clear feedback and hard rules mattered more than clever prompts.
6. **Make the pipeline visible.** Showing the scores and the prompt next to each answer made every bug easy to locate.

## Exercises to try

- **Wire in reflection.** Call `reflect()` on a few anchors (her career, her family, setbacks), save, and see how her reflective answers change.
- **Make recency real.** Advance the time step with each question and pass `stateless=False`, so recently discussed memories stay fresh across an interview.
- **Tune the weights.** Try relevance at 1 instead of 3. Do her most important memories start crowding out the right ones?
- **Remember the interview.** Store each question and answer as a new memory, so Trang can refer back to what you asked earlier.
- **Give the baker a real memory.** Replace Part 1's heading-based recipe lookup with this retrieval, and let her remember every cake she bakes.

## Try it yourself

Both projects run on your laptop with Ollama. Pick one and give it a try:

- **Talk to Trang:** [Sim-Intelligence-Lab/sim\_interview\_agent](https://github.com/Sim-Intelligence-Lab/sim_interview_agent)
- **Trang in the kitchen:** [Sim-Intelligence-Lab/sim\_baking\_agent](https://github.com/Sim-Intelligence-Lab/sim_baking_agent)

Each README walks you through setup. Ask Trang a question, or press Play and watch her bake, and let me know what she gets right and what she gets wrong.
