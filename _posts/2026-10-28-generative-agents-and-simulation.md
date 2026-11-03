---
layout: post
title: "Generative AI Agents and Simulation"
summary: "Why researchers are building little towns full of AI people, how the agents inside them are wired, and how to tell whether any of it is trustworthy"
date: 2026-10-28 10:00:00
---

*Part 1 of 2 on generative agents. Part 2: [Building Trang](/posts/2026/10/28/building-trang/), the implementation.*

I've spent the last few months building a generative agent called Trang. She started as a baker in an 8×8 pixel kitchen and ended up as an environmental lawyer you can interview about her life. I had the architecture working long before I understood what it was *for*. Why simulate people at all? And how would you know if the answer was wrong?

This post is the ideas, and [Part 2](/posts/2026/10/28/building-trang/) is the implementation. Everything here is about **simulation agents**, which are built to imitate how people behave. **Task agents**, like coding assistants, are built to get work done. The two share most of their wiring, but they're judged by different standards.

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

## 4. How does a generative agent work?

### Good at a moment, bad at a day

Ask a language model what a law student would do at 8 a.m. on a Tuesday and you'll get a good answer. A simulation asks again at 8:15, at 8:30, and every fifteen minutes for a week, and that's where three things go wrong:

- **It forgets.** Every call starts from nothing. You can't paste a week of someone's life into every prompt.
- **It never steps back.** Even with the history in front of it, the model sees a list of events. It doesn't notice "I keep doing this, so it must matter to me."
- **It doesn't stick to anything.** Every answer is the most plausible thing *right now*. Checking email at 9:00 makes sense, and at 9:20 it still makes sense, so the agent checks email all morning.

A **generative agent** is a language model plus one fix for each problem: a **memory** it can search, **reflection** so it can draw conclusions, and **planning** so its actions hold together over a day. The design comes from the 2023 paper of the same name, which ran 25 of these agents in a small simulated town.

### What it looks like from the outside

Our example agent is a law student, and all we write is one paragraph about who she is. From that, she plans her day: study group at 8, the law clinic from 1 to 4. A classmate mentions a talk on Friday, so she adds it to her plan. On Thursday, asked about her week, she brings up the talk and who told her about it. None of that is scripted. It all comes from the loop below.

### Everything is a memory

The agent keeps one long list called the **memory stream**. Each entry is a plain-English sentence with a timestamp:

```
08:02  reviews a case file
08:10  study group meets
13:00  plan: law clinic hours, 180 minutes
13:04  drafting a case memo
```

What she sees, does, plans and concludes all go into this one list, in plain English the model can read directly.

### Memory: find the few entries that matter

**The problem.** After a few days the list holds thousands of entries. You can't send them all. If you squash them into one summary, the details wash out, and "What have you been up to?" gets a vague answer like "classes and errands."

**The fix.** Score every memory against the current question, and send only the top few. The score adds three things:

```
score = recency + importance + relevance
```

<figure class="diagram">
{% include diagrams/ga-retrieval.svg %}
</figure>

- **Recency:** was it used recently? Every memory fades a little each hour (×0.995), and using it makes it fresh again.
- **Importance:** how big a deal was it? The model rates each memory from 1 to 10 when it's saved. Brushing teeth is a 2; a break-up is a 9.
- **Relevance:** is it about the question? This compares the meaning of the memory and the question (cosine similarity of their embeddings).

Each score is scaled to 0–1 and the three are added with equal weight. Ask "What did you do at the clinic?" and you get the clinic memory back, not her childhood weekends in the forest.

### Reflection: turn events into beliefs

**The problem.** Memories are only events. Ask "What do you care about?" and the agent reads off what it did lately: "I reviewed a case file and went to study group." True, but no person answers that way.

**The fix.** Every so often the agent stops to think. When the importance of her recent memories adds up past a threshold (a few times a day), she looks back over them and asks a few questions about herself. She pulls the matching memories and writes down a conclusion along with its evidence:

```
reflection: I believe in grassroots work
  because: volunteers with a neighborhood group, picked the clinic's
  local water case, stayed late to help a tenant
```

Reflections go back into the memory stream, so retrieval can find them and later reflections can build on them: events at the bottom, beliefs about herself on top. Now "What do you care about?" gets a real answer.

### Planning: make the day hold together

**The problem.** Without a plan, every moment is decided on its own, which is how you get the agent that checks email all morning.

**The fix.** Plan from the top down. Each morning the model sketches the day in 5–8 chunks: study group, lunch, clinic, gym, dinner. It then breaks each chunk into hours, and each hour into 5–15 minute steps. Plans are saved as memories too.

**Reacting.** A plan isn't a cage. At every step the agent asks, "Given what I just saw, should I keep going or react?" If a client walks into the clinic upset, she stops drafting the memo and talks to them. The rest of her plan is then rewritten from that moment on.

### The whole loop

<figure class="diagram">
{% include diagrams/ga-architecture.svg %}
<figcaption>Perceive, save to memory, retrieve what matters, act. Reflection and planning both read from memory and write back to it.</figcaption>
</figure>

The idea underneath every step is that **everything is a memory**. Observations, plans and reflections all live in one list, and every action is built from whatever retrieval pulls out. That gives you two useful side effects:

- **You can see why.** When an agent does something strange, print the memories it retrieved. The reason is usually right there.
- **Social behavior comes free.** Put several agents in one town and they pass along news, remember each other and make plans together, because a conversation is just one more thing they perceive and remember.

The cost is the number of model calls: agents × time steps × calls per step. A town living through a few days makes a lot of calls.

## 5. Is it any good? Believable versus accurate

You've built an agent. How do you know whether it works? That depends on what it's for, and it helps to separate two goals that get conflated.

### Believable agents

Disney animators Frank Thomas and Ollie Johnston called their craft [*the illusion of life*](https://en.wikipedia.org/wiki/The_Illusion_of_Life) (1981): characters who seem to think, decide, and act on their own. In 1994, AI researcher Joseph Bates borrowed the idea as the [**believable agent**](https://doi.org/10.1145/176789.176803), the interactive cousin of a believable character in a film. Believability is about whether the agent *feels alive*.

You test it the way the [Turing test](https://doi.org/10.1093/mind/LIX.236.433) does: do people buy it? The [generative agents paper](https://arxiv.org/abs/2304.03442) evaluated its agents this way, by interviewing them and having human raters judge the responses. It also ablated the architecture, removing reflection, planning, or observation memory in turn. Each removal measurably hurt believability, and the full architecture outscored human crowdworkers role-playing the same characters. Believable agents are what you want for games, storytelling, AI companions, and rehearsal spaces where people practice hard conversations, patient interviews, or conflict resolution.

Believable agents don't need to be smart. Reactivity, emotion, and social competence matter more than reasoning. But believability is only a convincing imitation, and a convincing character can still be wrong.

### Accurate agents

An **accurate** simulation is a prediction of the future. Most wicked problems in social science, urban planning, and market research need this, not a good story. The recipe is easy to state: gather ground truth and check whether the simulation reproduces it.

In practice the difficulty depends on the level:

- **Individuals:** what axis do you even evaluate on? And people disagree with themselves.
- **Groups:** the dynamics are complex, and they may not be predictable even in principle.
- **Populations:** ground truth is sometimes missing, and if you rerun a famous study, the model may just have memorized it.

That last problem led to a sharper test: can a model predict studies it *could not have seen*? A [2024 Stanford study](https://ai4pb.stanford.edu/projects/predicting-results-of-social-science-experiments-using-large-language-models) did exactly this. GPT-4 was asked to predict the outcomes of 70 pre-registered, nationally representative US survey experiments, 476 treatment effects in all, covering more than 100,000 participants. The predictions correlated with the real results at *r* = 0.85. On a held-out set of studies still unpublished at the time, and so not in any training data, it was *r* = 0.90. GPT-4 matched or beat human forecasters on the same task. That number surprised me more than anything else in this material.

For individuals, the most promising lever is **what you tell the agent about the person.** A one-line description like "35-year-old nurse from Ohio" gives the model little more than a stereotype, so you get the average of everyone who fits it. In a [2024 study](https://arxiv.org/abs/2411.10109) that built agents for about 1,000 real people, agents given each person's own two-hour interview matched that person's survey answers better than agents given only their demographics. That's the same memory stream as above, filled with a real person's words instead of made-up ones.

So, as of now:

- **Population level:** promising, even where memorization is ruled out.
- **Individual level:** early, but encouraging when agents are built from a person's own words.
- **Group level:** open, and maybe the real question.

### The part I'd underline

Believable doesn't mean accurate, and accurate doesn't require believable. A simulation can feel completely convincing and predict the wrong outcome. A highly accurate one may never need to show an emotion. Trang is firmly in the believable column: she answers in character, from the memories I gave her, and I have no evidence at all that she'd predict what a real person with that history would say. Before you judge an agent, decide which of the two you need.

<figure class="diagram">
{% include diagrams/ga-believable.svg %}
</figure>

## Where this leaves me

The question I keep coming back to is trust. When a simulation shows you something new, something no one predicted, how do you know whether to believe it? Schelling's model is trustworthy because its rules fit on an index card and you can read them. A generative agent has a billion-parameter model in the loop, and its "rules" are whatever the training data taught it. I don't have a good answer. The closest I've got is the panel under every one of Trang's answers: if I can see which memories she retrieved and what the prompt said, I can at least tell *why* she said something, even when I can't tell whether a real person would have.

Next post: the implementation. [Building Trang](/posts/2026/10/28/building-trang/) implements the memory, retrieval and reflection from section 4 on a 4B-parameter local model, no API key, with the retrieval scores visible under every answer.

## Further reading

- Thomas C. Schelling, [Dynamic Models of Segregation](https://doi.org/10.1080/0022250X.1971.9989794), *Journal of Mathematical Sociology* (1971), and *Micromotives and Macrobehavior* (1978)
- Horst Rittel and Melvin Webber, [Dilemmas in a General Theory of Planning](https://doi.org/10.1007/BF01405730), *Policy Sciences* (1973)
- Allen Newell, *Unified Theories of Cognition* (1990)
- Alan Turing, [Computing Machinery and Intelligence](https://doi.org/10.1093/mind/LIX.236.433), *Mind* (1950)
- Frank Thomas and Ollie Johnston, *The Illusion of Life: Disney Animation* (1981)
- Joseph Bates, [The Role of Emotion in Believable Agents](https://doi.org/10.1145/176789.176803), *Communications of the ACM* (1994)
- [Social Simulacra](https://arxiv.org/abs/2208.04024) (UIST 2022) and [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) (UIST 2023); the [reference code](https://github.com/joonspk-research/generative_agents) is open source
- [Generative Agent Simulations of 1,000 People](https://arxiv.org/abs/2411.10109) (2024)
- Theodore Sumers, Shunyu Yao, Karthik Narasimhan and Thomas Griffiths, [Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427), *TMLR* (2024), the general framework that generative agents are one instance of
- Luke Hewitt, Ashwini Ashokkumar, Isaias Ghezae and Robb Willer, [Predicting Results of Social Science Experiments Using Large Language Models](https://ai4pb.stanford.edu/projects/predicting-results-of-social-science-experiments-using-large-language-models) (2024 working paper)
