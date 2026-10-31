---
layout: post
title: "Generative Agents and Simulation"
summary: "Why researchers are building little towns full of AI people, how the agents inside them are wired, and how to tell whether any of it is trustworthy"
date: 2026-10-28 10:00:00
---

*Part 1 of 2 on generative agents. Part 2: [Building Trang](/posts/2026/10/28/building-trang/), an implementation of a generative agent.*

I've spent the last few months building a generative agent called Trang. She started as a baker in an 8×8 pixel kitchen and ended up as an environmental lawyer you can interview about her life. I had the architecture working long before I understood what it was *for*. Why simulate people at all? And how would you know if the answer was wrong?

This post covers the ideas, and [Part 2](/posts/2026/10/28/building-trang/) is the implementation of a generative agent. Everything here is about **simulation agents**, which are built to imitate how people behave. **Task agents**, like coding assistants, are built to get work done. The two share most of their wiring, but they're judged by different standards.

## 1. What is a simulation?

A simulation is a program that defines an **environment** and the **behaviors of individuals**, then shows you the world that results. *The Sims* is one. The forest-fire model from a science class is another. Every one of them has the same two ingredients:

- **The environment:** its current state, plus the rules for how it changes.
- **The agents:** the individuals living in it, each with their own state and behavior.

The whole idea fits in one loop. The world at tick *t* is the environment plus every agent. To get the world at tick *t+1*, apply the environment's rules and each agent's behavior to the world at *t*. Feed the result back in and go again.

<figure class="diagram">
{% include diagrams/ga-simulation-loop.svg %}
<figcaption>Each tick, the environment's rules and every agent's behavior turn the current world into the next one.</figcaption>
</figure>

It's small enough to write out in full:

```python
def run(world, ticks):
    for t in range(ticks):
        world.env.apply_rules()          # clocks advance, objects change state
        for agent in world.agents:       # order matters; randomize or round-robin
            obs = agent.perceive(world)  # what can it see from where it stands?
            action = agent.act(obs)      # the interesting part (section 4)
            world.apply(agent, action)   # move, speak, change an object
        yield world
```

Trang's first version was exactly this loop with one agent: perceive the kitchen, decide, act, let a rulebook apply the action, repeat. The first thing I learned from it is that `world.apply` is not optional. For a while her actions went through unchecked and the kitchen never told her no, and I watched the flour in her bowl climb to 3,325 grams out of a recipe that called for 500.

Three things make this loop different from an ordinary program:

1. **You can rerun it.** Start from the same state and go again. Once `agent.act` has an LLM in it the step is stochastic, so you get a different world each run. That's a feature (you can estimate a distribution) and a cost (one run tells you almost nothing, which I learned the hard way when Trang baked a perfect cake once and I assumed she was fixed).
2. **You can intervene.** Change one rule, hold everything else fixed, see what follows. It's a controlled experiment on a world you own.
3. **It surprises you.** You wrote every rule, and the outcome still isn't obvious. Simple parts interacting over many ticks produce patterns nobody wrote down. Schelling's segregation model is the textbook case: households that are happy with only 30% similar neighbors sort themselves into strongly segregated neighborhoods that none of them asked for.

### Why bother

Because some questions can't be tested in the real world. You can't rerun a pandemic with a different policy. A simulation lets you build a multiverse of possibilities and compare them side by side.

### This is not new

Researchers were doing this long before generative AI:

- **Cellular automata:** grids of simple cells following simple rules.
- **Game theory:** agents choosing strategies against each other.
- **Agent-based models**, such as Schelling's.

There's even a theory in cognitive science, *simulation theory of mind*, that says this is how people understand each other: you predict someone by running a small model of them in your head. I don't know how seriously psychologists take it anymore, but it's a nice frame for what follows.

The old models are simple and you can read every rule. Their weakness is that they flatten people. An agent whose whole personality is "if fewer than 30% of my neighbors are like me, move" is one thin slice of a human.

### What changed

You can hand a language model a line as short as "Trang is an environmental lawyer who grew up exploring forests with her grandmother" and get human-like behavior out, drawn from an enormous range of written experience. That's the engine behind **generative agents**. They're open-ended and capture nuance that no rule set could, and they're much harder to build and much harder to trust. Most of what follows is about that trade.

## 2. Why simulate? Wicked problems

Simulation is for a particular kind of problem. Social media platforms built attention-maximizing feeds to grow, then added moderation to cope with what spread, then got hit with bias and censorship complaints over the moderation. Pandemic lockdowns slowed spread less than hoped and hurt the economy; relaxing them under shifting guidance burned public trust, which came back to bite vaccination. Every fix produced the next problem.

<figure class="diagram">
{% include diagrams/ga-wicked-problem.svg %}
</figure>

Design theorists named this in the 1970s: **wicked problems**. Three of their properties matter here.

- **There's no finish line.** You can't even state the problem precisely, so you never know you're done.
- **Answers aren't right or wrong**, only better or worse, and there's no quick test.
- **You only get one shot.** Every real attempt changes the world, so you can't learn by trial and error.

The last one is the whole argument for simulation. If reality gives you one attempt, a place to practice is worth a lot.

## 3. What do you simulate? Individuals, groups, populations

Before building anything, decide what level you're asking about. The individual agent is always the building block, but the *question* lives at one of three levels, and each has its own recipe and its own way of failing.

<figure class="diagram">
{% include diagrams/ga-levels.svg %}
</figure>

**Individuals.** One specific person with their own traits, beliefs, and history. The questions are narrow: would *this* person like these search results? How would they react to this treatment? The problem is noise. People are inconsistent, and the data on any one person is thin. Nobody has a settled recipe here; the best starting point today is a detailed persona plus a memory of that person's history, which is what Trang's second version is. Having built one, I'd add: the persona gets you the voice, the memory gets you the facts, and neither gets you consistency across two different questions about the same thing.

**Groups.** People who interact and influence each other. How might two people resolve a conflict? Can a team of crowdworkers cooperate? You get this by giving several individual agents memory and letting them run against each other over ticks. The problem is that group dynamics may never be fully predictable, and some researchers suspect they aren't, even in principle.

**Populations.** Many people who share attributes, studied as aggregates. Does one party prefer a policy more than another? You sample personas from demographic distributions and average the responses. The problem is that aggregates can't see individuals, and they happily reproduce whatever bias and stereotype the model learned.

The way I keep them straight: **group models are about interactions between people; population models are about interventions on the whole.**

Most generative-AI simulation research right now is at the population level, for two reasons that are a bit unflattering. It's the easiest to check, because you can rerun a published study and compare. And it dodges the open problem of modeling one specific person. In one sentence: population models are too coarse, individual models are too noisy, and group models may never be predictable. Pick the level that matches your question, and evaluate at that level.

## 4. How is an AI agent built?

How do you turn a language model into a character that stays coherent over days of simulated time? A bare model can't. It has no memory of yesterday, and its context window won't hold two days of experience. Cognitive scientists have a name for the missing piece: a **cognitive architecture**, the fixed machinery underneath thought, such as memory, goals, and decision-making. SOAR (1983) and ACT-R (1993) were built to model human problem solving. The **generative agent** architecture from the 2023 paper of that name revives the idea with an LLM in the middle. The paper demonstrates it on a town of 25 agents. The examples below use Trang, the agent I build in the next post, an environmental lawyer with 91 memories of her life.

<figure class="diagram">
{% include diagrams/ga-architecture.svg %}
<figcaption>Left to right: perception feeds the memory stream, retrieval pulls a small working set into scratch memory, and that set drives action. Plans and reflections loop back in as new memories.</figcaption>
</figure>

**1. Perceive and remember.** The agent records everything it notices ("Trang is reviewing a permit case at the clinic") in its **memory stream**, an append-only list of plain-language records. Each one carries its text, when it was created, when it was last accessed, an importance score, and an embedding. The stream gets far too long to hand to the model whole, and dumping it all in would drown the signal anyway. So the agent has to choose what to recall.

**2. Retrieve.** Retrieval scores every record against the current situation and takes the top of the list:

```
score(m) = α_rec · recency(m) + α_imp · importance(m) + α_rel · relevance(m, q)
```

<figure class="diagram">
{% include diagrams/ga-retrieval.svg %}
</figure>

The three terms, as the paper implements them:

- **Recency** is an exponential decay over simulated hours since the memory was *last accessed*, not created, with a factor of 0.995 per hour. Touching a memory refreshes it, so things the agent keeps thinking about stay warm.
- **Importance** is one integer the LLM assigns when the record is written, by rating how poignant the event is. Brushing teeth near the bottom, a breakup near the top.
- **Relevance** is cosine similarity between the record's embedding and the query's, from the same embedding model.

Each term is min-max normalized to [0, 1] across the candidates before summing, and the paper sets all three weights to 1. The sorted list gets truncated to whatever fits the prompt's token budget, and every record that makes the cut has its last-accessed time refreshed.

> **From building Trang.** The code I started Trang from rates importance 0–100, and I'd assumed that was the paper's scale too. It isn't; the paper uses 1–10. I spent a while wondering whether I'd broken something before noticing that the normalization step makes the scale irrelevant: 1–10 and 0–100 land in the same [0, 1] range before they're summed. What *does* matter is giving the model anchors in the prompt ("routine events 0–15, defining life events 70–100"), because without them a small model rates everything in the sixties.
>
> The weights matter more than the scale. For an interview agent that answers one question at a time, I ended up at 3 · relevance + 0.5 · importance + 0.5 · recency, because being on topic beats everything else, with decay per time step rather than per hour. The best answer to Trang's opening interview question came from a memory rated 30 out of 100; 74 of her 91 memories outrank it on importance. It won because it was exactly on topic. With the paper's equal weights, I'm fairly sure it would have lost to a childhood memory about the forest. The next post has the full ranking table.

Ask Trang "What have you worked on in your law school's Environmental Law Clinic?" and retrieval surfaces the clinic memory, not her childhood weekends in the forest.

**3. Act.** The retrieved records land in **scratch memory**, the agent's working set. The prompt is scratch memory plus a summary of who the agent is and what it's doing right now, and the LLM decides: say something, walk somewhere, or change the world, like pulling up a case file.

**4. Reflect.** Observations alone don't let an agent generalize about itself. So periodically it reviews its hundred or so most recent records and writes higher-level thoughts, such as "Losing the city council race deepened my commitment to grassroots activism." The mechanics: reflection fires when the summed importance of recent events crosses a threshold (150 in the paper, so a few times a day). The agent asks the LLM for the three most salient questions those records raise, retrieves against each question, and asks for insights that cite the supporting records. The insights go into the stream as new records, which means later reflections can build on earlier ones. Over time you get a tree: observations at the leaves, increasingly abstract thoughts higher up.

This is the part of Trang I haven't wired into the app yet. The function exists, and when I ran it by hand on "what has losing the city council race taught me?" (she ran for city council in her backstory, and lost), she produced a belief about public service that no single memory states. The model rated those insights higher than almost all her raw memories, which is the point: reflections are what should win retrieval when someone asks a big question.

**5. Plan.** Sensible moment-to-moment behavior isn't enough; an agent should also make sense across a day. Planning is top-down and recursive. The LLM drafts a rough day in five to eight chunks (wake at 7, clinic at 9, study group from 2 to 5), then expands each chunk into hourly steps, then into five-to-fifteen-minute actions with a location, a start time, and a duration. Plans are stored as memories too. Each tick the agent checks whether what it just saw calls for a reaction, and if so it re-plans from that point rather than following the stale schedule.

The idea underneath all five steps is that **everything is a memory**. Perceptions, plans, and reflections go into one stream, and everything the agent does is built from what it reads back out. That's what keeps behavior consistent over time, and it's what makes the thing debuggable: when an agent does something strange, print the records it retrieved and you can usually see why. Trang's interview app shows the retrieved memories and their scores under every answer for exactly this reason; it's the single feature that made the project tolerable to work on. It's also what makes the architecture expensive. Cost is agents × ticks × LLM calls per tick, and a tick can involve perception, retrieval scoring, action generation, and the occasional reflection and planning pass.

## 5. Is it any good? Believable versus accurate

You've built an agent. How do you know whether it works? That depends on what it's for, and it helps to separate two goals that get conflated.

### Believable agents

Disney's animators described their craft as the *illusion of life*: characters who seem to think, decide, and act of their own volition. AI researchers borrowed the idea in the 1990s as the **believable agent**, the interactive cousin of a believable character in a film. Believability is about whether the agent *feels alive*.

You test it the way the Turing test does: do people buy it? The generative agents paper evaluated its agents this way, by interviewing them and having human raters judge the responses. It also ablated the architecture, removing reflection, planning, or observation memory in turn. Each removal measurably hurt believability, and the full architecture outscored human crowdworkers role-playing the same characters. Believable agents are what you want for games, storytelling, AI companions, and rehearsal spaces where people practice hard conversations, patient interviews, or conflict resolution.

Believable agents don't need to be smart. Reactivity, emotion, and social competence matter more than reasoning. But believability is only a convincing imitation, and a convincing character can still be wrong.

### Accurate agents

An **accurate** simulation is a prediction of the future. Most wicked problems in social science, urban planning, and market research need this, not a good story. The recipe is easy to state: gather ground truth and check whether the simulation reproduces it.

In practice the difficulty depends on the level:

- **Individuals:** what axis do you even evaluate on? And people disagree with themselves.
- **Groups:** the dynamics are complex, and some researchers think prediction here is impossible in principle.
- **Populations:** ground truth is sometimes missing, and if you rerun a famous study, the model may just have memorized it.

That last problem led to a sharper test: can a model predict studies it *could not have seen*? A Stanford study, first released in 2024, did exactly this. GPT-4 was asked to predict the outcomes of 70 pre-registered, nationally representative US survey experiments, 476 treatment effects in all, covering more than 100,000 participants. The predictions correlated with the real results at *r* = 0.85. On a held-out set of studies still unpublished at the time, and so not in any training data, it was *r* = 0.90. GPT-4 matched or beat human forecasters on the same task. That number surprised me more than anything else in this material.

So, as of now:

- **Population level:** promising, even where memorization is ruled out.
- **Individual level:** the verdict is still out.
- **Group level:** open, and maybe the real question.

### The part I'd underline

Believable doesn't mean accurate, and accurate doesn't require believable. A simulation can feel completely convincing and predict the wrong outcome. A highly accurate one may never need to show an emotion. Trang is firmly in the believable column: she answers in character, from the memories I gave her, and I have no evidence at all that she'd predict what a real person with that history would say. Before you judge an agent, decide which of the two you need.

<figure class="diagram">
{% include diagrams/ga-believable.svg %}
</figure>

## Where this leaves me

The question I keep coming back to is trust. When a simulation shows you something new, something no one predicted, how do you know whether to believe it? Schelling's model is trustworthy because its rules fit on an index card and you can read them. A generative agent has a billion-parameter model in the loop, and its "rules" are whatever the training data taught it. I don't have a good answer. The closest I've got is the panel under every one of Trang's answers: if I can see which memories she retrieved and what the prompt said, I can at least tell *why* she said something, even when I can't tell whether a real person would have.

Next post: the implementation. [Building Trang](/posts/2026/10/28/building-trang/) implements the architecture in section 4 on a 4B-parameter local model, no API key, with the retrieval scores visible under every answer.

## Further reading

- Thomas C. Schelling, [Dynamic Models of Segregation](https://doi.org/10.1080/0022250X.1971.9989794), *Journal of Mathematical Sociology* (1971), and *Micromotives and Macrobehavior* (1978)
- Horst Rittel and Melvin Webber, [Dilemmas in a General Theory of Planning](https://doi.org/10.1007/BF01405730), *Policy Sciences* (1973)
- Allen Newell, *Unified Theories of Cognition* (1990)
- Joseph Bates, [The Role of Emotion in Believable Agents](https://doi.org/10.1145/176789.176803), *Communications of the ACM* (1994)
- Joon Sung Park et al., [Social Simulacra](https://arxiv.org/abs/2208.04024) (UIST 2022) and [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) (UIST 2023); the [reference code](https://github.com/joonspk-research/generative_agents) is open source
- Theodore Sumers, Shunyu Yao, Karthik Narasimhan and Thomas Griffiths, [Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427), *TMLR* (2024), the general framework that generative agents are one instance of
- Luke Hewitt, Ashwini Ashokkumar, Isaias Ghezae and Robb Willer, [Predicting Results of Social Science Experiments Using Large Language Models](https://ai4pb.stanford.edu/projects/predicting-results-of-social-science-experiments-using-large-language-models) (2024 preprint; since published in *Nature*)
