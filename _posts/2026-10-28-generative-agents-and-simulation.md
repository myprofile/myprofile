---
layout: post
title: "Generative Agents and Simulation"
summary: "Why researchers are building little towns full of AI people, how the agents inside them are wired, and how to tell whether any of it is trustworthy"
date: 2026-10-28 18:00:00
---

I've spent the last few months building a generative agent called Trang. She started as a baker in an 8×8 pixel kitchen and ended up as an environmental lawyer you can interview about her life. Along the way I had the architecture before I had the reasons for it: I knew an agent needs a memory stream, a retrieval function, and a reflection step, because the paper said so and the code worked. What I didn't have was a clear picture of what the whole thing was *for*. Why simulate people at all? What question is a town of 25 AI residents supposed to answer? And how would you know if the answer was wrong?

This post is the ideas. The next one is the code. Here: what a simulation is, which problems it's for, what you choose to simulate, how an agent is wired, and how to judge one. Next: building Trang's memory stream, retrieval, and reflection from scratch on a local model, where every number in section 4 below shows up in Python. I'll point at Trang a few times along the way, because the ideas made more sense to me once something concrete had gone wrong.

One distinction first, because it decides what this post is about. **Task agents** (coding assistants, travel bookers, AutoGPT-style tools) are built to get work done. **Simulation agents** are built to imitate and predict how people behave. They share most of their architecture and the two camps borrow from each other constantly, but they're judged by different standards. Trang is a simulation agent, and so is everything below.

**The short version**

1. A simulation is an environment plus agents, stepped forward one tick at a time.
2. It earns its keep on *wicked problems*, where reality gives you one attempt.
3. Pick your level of analysis before you build: individual, group, or population.
4. A generative agent is an LLM wrapped in a memory stream with retrieval, reflection, and planning. Retrieval scores `recency + importance + relevance`, each normalized to [0, 1].
5. Judge it by the right standard: believable (feels alive) or accurate (predicts correctly). These are different things, and the evidence for each sits at different levels.

## 1. What is a simulation?

A simulation is a program that defines an **environment** and the **behaviors of individuals**, then shows you the world that results. *The Sims* is one. The forest-fire model from a science class is another. Every one of them has the same two ingredients:

- **The environment:** its current state, plus the rules for how it changes.
- **The agents:** the individuals living in it, each with their own state and behavior.

The whole idea fits in one loop. The world at tick *t* is the environment plus every agent. To get the world at tick *t+1*, apply the environment's rules and each agent's behavior to the world at *t*. Feed the result back in and go again.

![Simulation loop: the world at tick t (environment state plus agents) passes through step(), which applies the environment rules and lets each agent perceive and act, producing the world at tick t+1, which is fed back in as the next tick.](/assets/img/generative-agents/simulation-loop.png)

*Each tick, the environment's rules and every agent's behavior turn the current world into the next one.*

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

1. **You can rerun it.** Start from the same state and go again. Once `agent.act` has an LLM in it the step is stochastic, so you get a different world each run. That's a feature (you can estimate a distribution) and a cost (one run tells you almost nothing, which I [[learned the hard way when Trang baked a perfect cake once and I assumed she was fixed]]).
2. **You can intervene.** Change one rule, hold everything else fixed, see what follows. It's a controlled experiment on a world you own.
3. **It surprises you.** You wrote every rule, and the outcome still isn't obvious. Simple parts interacting over many ticks produce patterns nobody wrote down. Schelling's segregation model is the textbook case: households that are happy with only 30% similar neighbors sort themselves into near-total segregation that none of them asked for.

### Why bother

Because some questions can't be tested in the real world. You can't rerun a pandemic with a different policy. A simulation lets you build a multiverse of possibilities and compare them side by side.

### This is not new

Researchers were doing this long before generative AI:

- **Cellular automata:** grids of simple cells following simple rules.
- **Game theory:** agents choosing strategies against each other.
- **Agent-based models**, such as Schelling's.

There's even a theory in cognitive science, *simulation theory of mind*, that says this is how people understand each other: you predict someone by running a small model of them in your head. [[I don't know how seriously psychologists take it anymore, but]] it's a nice frame for what follows.

The old models are simple and you can read every rule. Their weakness is that they flatten people. An agent whose whole personality is "if fewer than 30% of my neighbors are like me, move" is one thin slice of a human.

### What changed

You can hand a language model a line as short as "Isabella owns a café and is planning a party" and get human-like behavior out, drawn from an enormous range of written experience. That's the engine behind **generative agents**. They're open-ended and capture nuance that no rule set could, and they're much harder to build and much harder to trust. Most of what follows is about that trade.

## 2. Why simulate? Wicked problems

Simulation is for a particular kind of problem. Social media platforms built attention-maximizing feeds to grow, then added moderation to cope with what spread, then got hit with bias and censorship complaints over the moderation. Pandemic lockdowns slowed spread less than hoped and hurt the economy; relaxing them under shifting guidance burned public trust, which came back to bite vaccination. Every fix produced the next problem.

![Wicked problem loop: intervene → the world changes → a new problem appears → intervene again, with the note that in reality you cannot rewind. Beside it, simulation breaks the loop: rerun from the same state, intervene on one rule, compare the worlds side by side.](/assets/img/generative-agents/wicked-problem-loop.png)

Design theorists named this in the 1970s: **wicked problems**. Three of their properties matter here.

- **There's no finish line.** You can't even state the problem precisely, so you never know you're done.
- **Answers aren't right or wrong**, only better or worse, and there's no quick test.
- **You only get one shot.** Every real attempt changes the world, so you can't learn by trial and error.

The last one is the whole argument for simulation. If reality gives you one attempt, a place to practice is worth a lot.

## 3. What do you simulate? Individuals, groups, populations

Before building anything, decide what level you're asking about. The individual agent is always the building block, but the *question* lives at one of three levels, and each has its own recipe and its own way of failing.

![Three panels comparing levels of analysis. Individual: one specific person; question "would this person like these search results?"; modeled with a detailed persona and memory; fails on noise. Group: a few people who interact; question "how would these two resolve this conflict?"; modeled as several memory-equipped agents interacting over ticks; dynamics may never be fully predictable. Population: many people aggregated; question "do older adults read more than younger ones?"; modeled by sampling personas from demographics and averaging; too coarse, can reproduce bias. A double arrow underneath: trade detail for scale.](/assets/img/generative-agents/levels-of-analysis.png)

**Individuals.** One specific person with their own traits, beliefs, and history. The questions are narrow: would *this* person like these search results? How would they react to this treatment? The problem is noise. People are inconsistent, and the data on any one person is thin. Nobody has a settled recipe here; the best starting point today is a detailed persona plus a memory of that person's history, which is what Trang's second version is. [[Having built one, I'd add: the persona gets you the voice, the memory gets you the facts, and neither gets you consistency across two different questions about the same thing.]]

**Groups.** People who interact and influence each other. How might two people resolve a conflict? Can a team of crowdworkers cooperate? You get this by giving several individual agents memory and letting them run against each other over ticks. The problem is that group dynamics may never be fully predictable, and some researchers suspect they aren't, even in principle.

**Populations.** Many people who share attributes, studied as aggregates. Does one party prefer a policy more than another? You sample personas from demographic distributions and average the responses. The problem is that aggregates can't see individuals, and they happily reproduce whatever bias and stereotype the model learned.

The way I keep them straight: **group models are about interactions between people; population models are about interventions on the whole.**

Most generative-AI simulation research right now is at the population level, for two reasons that are a bit unflattering. It's the easiest to check, because you can rerun a published study and compare. And it dodges the open problem of modeling one specific person. In one sentence: population models are too coarse, individual models are too noisy, and group models may never be predictable. Pick the level that matches your question, and evaluate at that level.

## 4. How is an AI agent built?

How do you turn a language model into a character that stays coherent over days of simulated time? A bare model can't. It has no memory of yesterday, and its context window won't hold two days of experience. Cognitive scientists have a name for the missing piece: a **cognitive architecture**, the fixed machinery underneath thought, such as memory, goals, and decision-making. SOAR and ACT-R were built in the 1980s to model human problem solving. The **generative agent** architecture from the 2023 paper of that name revives the idea with an LLM in the middle. The examples below come from its reference implementation, a town of 25 agents where one resident, Isabella, runs a café and is planning a party.

![Generative agent architecture: the environment feeds perceive, which writes to an append-only memory stream of timestamped natural-language records (observations, plans, reflections). Retrieve pulls the top-k records by score into scratch memory, the working set that fits in the prompt; the LLM turns that into an act, which changes the environment. Reflect reads about a hundred recent records and writes insights back to the stream; plan writes day, hour, and 5–15-minute plans back to the stream too.](/assets/img/generative-agents/generative-agent-architecture.png)

*Left to right: perception feeds the memory stream, retrieval pulls a small working set into scratch memory, and that set drives action. Plans and reflections loop back in as new memories.*

**1. Perceive and remember.** The agent records everything it notices ("Maria is chatting with Klaus") in its **memory stream**, an append-only list of plain-language records. Each one carries its text, when it was created, when it was last accessed, an importance score, and an embedding. The stream gets far too long to hand to the model whole, and dumping it all in would drown the signal anyway. So the agent has to choose what to recall.

**2. Retrieve.** Retrieval scores every record against the current situation and takes the top of the list:

```
score(m) = α_rec · recency(m) + α_imp · importance(m) + α_rel · relevance(m, q)
```

![Retrieval scoring: a query ("What are you excited about?") and a memory ("ordering decorations for the party") feed three scorers. Recency: 0.995 to the power of hours since last accessed, drawn as a decay curve. Importance: a 1–10 rating the LLM assigns once when the memory is written (brushing teeth 2, break-up 9). Relevance: cosine similarity of the two embeddings. Each is min-max normalized to [0, 1], summed with weights of 1, sorted, and the top memories are taken until the prompt's token budget is full.](/assets/img/generative-agents/retrieval-scoring.png)

The three terms, as the paper implements them:

- **Recency** is an exponential decay over simulated hours since the memory was *last accessed*, not created, with a factor of 0.995 per hour. Touching a memory refreshes it, so things the agent keeps thinking about stay warm.
- **Importance** is one integer the LLM assigns when the record is written, by rating how poignant the event is. Brushing teeth near the bottom, a breakup near the top.
- **Relevance** is cosine similarity between the record's embedding and the query's, from the same embedding model.

Each term is min-max normalized to [0, 1] across the candidates before summing, and the paper sets all three weights to 1. The sorted list gets truncated to whatever fits the prompt's token budget, and every record that makes the cut has its last-accessed time refreshed.

A detour on importance, because it tripped me up. The code I started Trang from rates importance 0–100, and I'd assumed that was the paper's scale too. It isn't; the paper uses 1–10. I spent a while wondering whether I'd broken something before noticing that the normalization step makes the scale irrelevant: 1–10 and 0–100 land in the same [0, 1] range before they're summed. What *does* matter is giving the model anchors in the prompt ("routine events 0–15, defining life events 70–100"), because without them a small model rates everything in the sixties.

The weights matter more than the scale. For an interview agent that answers one question at a time, I ended up at 3 · relevance + 0.5 · importance + 0.5 · recency, because being on topic beats everything else, with decay per time step rather than per hour. The best answer to Trang's opening interview question came from a memory rated 30 out of 100; 74 of her 91 memories outrank it on importance. It won because it was exactly on topic. With the paper's equal weights, [[I'm fairly sure]] it would have lost to a childhood memory about the forest. The next post has the full ranking table.

Ask Isabella "What are you excited about?" and retrieval surfaces her party plans and the decorations she ordered, not the piano someone played nearby last week.

**3. Act.** The retrieved records land in **scratch memory**, the agent's working set. The prompt is scratch memory plus a summary of who the agent is and what it's doing right now, and the LLM decides: say something, walk somewhere, or change the world, like turning on the coffee machine.

**4. Reflect.** Observations alone don't let an agent generalize about itself. So periodically it reviews its hundred or so most recent records and writes higher-level thoughts, such as "I enjoy organizing events and making people feel welcome." The mechanics: reflection fires when the summed importance of recent events crosses a threshold (150 in the paper, so a few times a day). The agent asks the LLM for the three most salient questions those records raise, retrieves against each question, and asks for insights that cite the supporting records. The insights go into the stream as new records, which means later reflections can build on earlier ones. Over time you get a tree: observations at the leaves, increasingly abstract thoughts higher up.

This is the part of Trang I haven't wired into the app yet. The function exists, and when I ran it by hand on "what has losing the city council race taught me?" (she ran for city council in her backstory, and lost), she produced a belief about public service that no single memory states. The model rated those insights higher than almost all her raw memories, which is the point: reflections are what should win retrieval when someone asks a big question.

**5. Plan.** Sensible moment-to-moment behavior isn't enough; an agent should also make sense across a day. Planning is top-down and recursive. The LLM drafts a rough day in five to eight chunks (wake at 8, class at 10, compose from 1 to 5), then expands each chunk into hourly steps, then into five-to-fifteen-minute actions with a location, a start time, and a duration. Plans are stored as memories too. Each tick the agent checks whether what it just saw calls for a reaction, and if so it re-plans from that point rather than following the stale schedule.

The idea underneath all five steps is that **everything is a memory**. Perceptions, plans, and reflections go into one stream, and everything the agent does is built from what it reads back out. That's what keeps behavior consistent over time, and it's what makes the thing debuggable: when an agent does something strange, print the records it retrieved and you can usually see why. Trang's interview app shows the retrieved memories and their scores under every answer for exactly this reason; [[it's the single feature that made the project tolerable to work on]]. It's also what makes the architecture expensive. Cost is agents × ticks × LLM calls per tick, and a tick can involve perception, retrieval scoring, action generation, and the occasional reflection and planning pass.

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

That last problem led to a sharper test: can a model predict studies it *could not have seen*? A 2024 study from Stanford did exactly this. GPT-4 was asked to predict the outcomes of 70 pre-registered, nationally representative US survey experiments, 476 treatment effects in all, covering more than 100,000 participants. The predictions correlated with the real results at *r* = 0.85. On a held-out set of studies still unpublished at the time, and so not in any training data, it was *r* = 0.90. GPT-4 matched or beat human forecasters on the same task. [[That number surprised me more than anything else in this material.]]

So, as of now:

- **Population level:** promising, even where memorization is ruled out.
- **Individual level:** the verdict is still out.
- **Group level:** open, and maybe the real question.

### The part I'd underline

Believable doesn't mean accurate, and accurate doesn't require believable. A simulation can feel completely convincing and predict the wrong outcome. A highly accurate one may never need to show an emotion. Trang is firmly in the believable column: she answers in character, from the memories I gave her, and I have no evidence at all that she'd predict what a real person with that history would say. Before you judge an agent, decide which of the two you need.

![Two-by-two grid of believability against accuracy. Convincing but wrong: feels alive, predicts the wrong outcome; fine for games, companions, rehearsal spaces; tested Turing-style, as the generative agents paper did. The goal: feels real and matches ground truth; nobody is reliably here yet. Neither: a bare LLM with no memory asked to be someone for three days. Accurate but plain: matches the data, shows no emotion; fine for social science, market research, urban planning; e.g. GPT-4 predicting 476 survey-experiment effects at r = 0.85, 0.90 on unpublished studies.](/assets/img/generative-agents/believable-vs-accurate.png)

## Where this leaves me

The question I keep coming back to is trust. When a simulation shows you something new, something no one predicted, how do you know whether to believe it? Schelling's model is trustworthy because its rules fit on an index card and you can read them. A generative agent has a billion-parameter model in the loop, and its "rules" are whatever the training data taught it. I don't have a good answer. The closest I've got is the panel under every one of Trang's answers: if I can see which memories she retrieved and what the prompt said, I can at least tell *why* she said something, even when I can't tell whether a real person would have.

Next post: the code. [Building Trang](/posts/2026/10/28/building-trang/) implements the architecture in section 4 on a 4B-parameter local model, no API key, with the retrieval scores visible under every answer.

## Further reading

- Thomas C. Schelling, [Dynamic Models of Segregation](https://doi.org/10.1080/0022250X.1971.9989794), *Journal of Mathematical Sociology* (1971), and *Micromotives and Macrobehavior* (1978)
- Horst Rittel and Melvin Webber, [Dilemmas in a General Theory of Planning](https://doi.org/10.1007/BF01405730), *Policy Sciences* (1973)
- Allen Newell, *Unified Theories of Cognition* (1990)
- Joseph Bates, [The Role of Emotion in Believable Agents](https://doi.org/10.1145/176789.176803), *Communications of the ACM* (1994)
- Joon Sung Park et al., [Social Simulacra](https://arxiv.org/abs/2208.04024) (UIST 2022) and [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) (UIST 2023); the [reference code](https://github.com/joonspk-research/generative_agents) is open source
- Theodore Sumers, Shunyu Yao, Karthik Narasimhan and Thomas Griffiths, [Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427), *TMLR* (2024), the general framework that generative agents are one instance of
- Luke Hewitt, Ashwini Ashokkumar, Isaias Ghezae and Robb Willer, [Predicting Results of Social Science Experiments Using Large Language Models](https://ai4pb.stanford.edu/projects/predicting-results-of-social-science-experiments-using-large-language-models) (2024)
