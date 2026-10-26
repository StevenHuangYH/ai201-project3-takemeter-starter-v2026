# TakeMeter

> ### 👋 Start here
>
> **New to this repo? Read [RUNNING.md](RUNNING.md) first** — setup, the
> notebook, the baseline, and what to do when something breaks.
>
> Once `python test.py` passes:
>
> ```bash
> head -5 data/practice_labels.csv     # the shape your labels.csv needs
> ```
>
> Then open `takemeter.ipynb` **in this folder** — in VS Code, or with
> `jupyter notebook` if you prefer. Pick the kernel: the `.venv` inside this
> project. Run section 1, which reports the hardware you'll be training on.
> Everything else waits until you have data.
>
> Nothing to upload, nothing to connect, no accounts and no keys. The notebook
> runs on your machine and writes next to your code.
>
> **The rest of this file is your submission.** Fill it in as you go.

---

<!-- ─────────────────────────────────────────────────────────────────────────
     HOW TO USE THIS FILE

     Unit 5 asks for the first five sections. Unit 6 adds the five below them.

     Everything is pasted as TEXT. No screenshots, no images.

     ⚠️ The confusion matrix especially. The notebook prints one as a markdown
     table, ready to copy. A screenshot of a matrix earns nothing. Paste the
     table.
     ───────────────────────────────────────────────────────────────────────── -->

<!-- ═══════════════════════ UNIT 5 — THE BUILD ═══════════════════════ -->

## What This Does

TakeMeter is a text classifier built to categorize posts from `r/nba`, one of the largest and most active sports discussion communities on the web. Basketball forums host a wide spectrum of discourse quality—ranging from deeply researched statistical breakdowns and salary cap analysis to wild, unsubstantiated hot takes and emotional game-thread venting.

TakeMeter automatically parses and classifies public community submissions into three distinct categories: `analysis`, `hot_take`, and `reaction`. By separating load-bearing empirical discussion from provocative opinions and visceral emotional reactions, the system can help filter noise, surface high-effort community contributions, and quantify sentiment dynamics across game days.

---

## Label Taxonomy

### `analysis`

**Definition:** Makes an argument backed by a specific, checkable fact such as a box score statistic, shooting percentage, contract number, historical record, or advanced metric that carries the claim.

**Example 1:**
> "In the 14 games without their starting center, their defensive rating plummeted from 108.4 (3rd in the league) to 119.2 (28th), while opponents shot 68% in the restricted area compared to 54% previously."

**Example 2:**
> "He is owed $48.7M next season with a 15% trade kicker, which means any team trading for him would hard-cap themselves at the second apron and lose their taxpayer mid-level exception."

### `hot_take`

**Definition:** A confident, controversial, or sweeping claim about a player, team, or narrative with no factual evidence or statistical support offered.

**Example 1:**
> "The 2017 Warriors would sweep any team in NBA history and it wouldn't even be competitive in the fourth quarter."

**Example 2:**
> "He has zero clutch gene and will never be the number one option on a championship team no matter who you put around him."

### `reaction`

**Definition:** An immediate, personal emotional response to a recent play, game result, or breaking news, capturing feelings of shock, joy, misery, or excitement rather than making an argument.

**Example 1:**
> "I am literally sick to my stomach watching this fourth quarter collapse, I had to turn the TV off."

**Example 2:**
> "HOLY SHIT WHAT A BUZZER BEATER! I'm screaming in my living room and I think I just woke up my entire apartment complex!"

### The hardest boundary

**Which two labels:** `analysis` vs. `hot_take`

**The decision rule I used every time:**
If the post names a specific checkable fact, it is `analysis` only if that fact is load-bearing and carries the argument. If a statistic or number is cited merely as rhetorical decoration or an afterthought within an otherwise sweeping, unsupported claim, the post is classified as `hot_take`.



---

## The Dataset

**Where the posts came from:**
Public submissions and discussion threads from `r/nba` (Reddit's primary NBA basketball community), gathered from game threads, post-game analytical discussions, and trade/offseason debate threads.

**How I labelled them:**
I labeled the first 20 posts completely cold without any AI assistance to establish an honest baseline and ground myself in the taxonomy. For the remaining 190 posts, I used an AI tool to pre-label candidate posts against the written definitions, followed by a manual review pass where I read, verified, and corrected every single row, adjudicating ambiguous edge cases using our decision rules. All rows are flagged in the `note` column as either `cold` or `pre-labeled, verified`.

**Counts per label:**

| Label | Count | Share |
|---|---|---|
| `analysis` | 70 | 33.3% |
| `hot_take` | 70 | 33.3% |
| `reaction` | 70 | 33.3% |
| **Total** | 210 | 100% |

**Three hard cases**

**1.**
> *The post:* "He scored 38 points last night on 14 shots and people are acting like he's an MVP candidate. One hot shooting night doesn't erase being a defensive traffic cone who gives up 40 on the other end."
>
> *Could have been:* `analysis` vs. `hot_take`
>
> *I chose, because:* I chose `hot_take`. Although the post cites concrete statistics (38 points on 14 shots), the numbers are purely ornamental and serve as rhetorical backdrop to an unsubstantiated sweeping assertion ("defensive traffic cone who gives up 40"). Under our decision rule, the fact does not carry the argument, so it remains a hot take with a statistic in it.

**2.**
> *The post:* "Up 5 with 18 seconds left and you commit a clear-path foul and turn the ball over on the inbounds. I have supported this franchise for 20 years and I have never felt this utterly betrayed."
>
> *Could have been:* `reaction` vs. `analysis`
>
> *I chose, because:* I chose `reaction`. While the post details the exact game score, remaining time, and procedural turnover sequence, these factual references merely provide context for the author's immediate personal anguish and emotional exhaustion ("never felt this utterly betrayed"). Under our decision rules, reporting immediate feelings outranks descriptive game recaps.

**3.**
> *The post:* "Stop calling this offense elite. Over the last 15 games they rank 24th in halfcourt offensive efficiency (91.2 points per 100) and turn the ball over on 16.8% of possessions against top-10 defenses. The transition numbers are completely masking their inability to run basic sets."
>
> *Could have been:* `hot_take` vs. `analysis`
>
> *I chose, because:* I chose `analysis`. Despite the aggressive rhetorical tone of the first sentence ("Stop calling this offense elite"), the argument is entirely load-bearing on verifiable empirical data (a 15-game sample, 91.2 halfcourt rating, 16.8% turnover rate). Under our decision rule, heated tone does not disqualify analysis when the facts carry the claim.

---

## The Training Run

<!-- Your starting model, your settings, and anything you changed and why. -->

**Base model:**

**Settings:** <!-- epochs, learning rate, batch size, seed -->

**Anything I changed from the defaults, and why:**

**Split sizes:** <!-- train / val / test, and per-label counts in the test
split. If a label had fewer than about 8 in test, say so — it explains a lot
of next unit's variance. -->



---

## How I Used AI

<!-- Two specific moments — what you asked, what came back, what you changed.

     ⚠️ Plus disclosure of any pre-labelling. If you had a model pre-label a
     batch and then read and corrected every one, say that. It's an allowed
     workflow and disclosing it costs you nothing. Not disclosing it is the
     problem. -->

**Moment 1**

- *What I asked for:*
- *What came back:*
- *What I changed:*

**Moment 2**

- *What I asked for:*
- *What came back:*
- *What I changed:*

**Pre-labelling disclosure:**

<!-- ═══════════════════════ UNIT 6 — THE TEST ═══════════════════════

     Don't fill these in during unit 5.
     ═══════════════════════════════════════════════════════════════════ -->

---

## Baseline vs. Trained

<!-- Both models on the same posts. `python baseline.py --trained results.json`
     prints this table for you. -->

| Measure | Baseline | Trained | Difference |
|---|---|---|---|
| Overall accuracy |  |  |  |
| Macro F1 |  |  |  |
| F1 — `label_one` |  |  |  |
| F1 — `label_two` |  |  |  |

**What I predicted before I looked:**
<!-- Milestone 1 asks you to write this BEFORE seeing the trained numbers. A
     prediction made afterwards isn't one. -->

**What the gap actually means:**
<!-- If the baseline matched your trained model, your fine-tuning added
     nothing — and that is a real finding, not a failure. Say it plainly. -->



---

## Run Log — Before

<!-- Five criteria across three seeds. The notebook's section 6 prints the
     spread table; the Target and Verdict columns are yours. -->

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1.  |  |  |  |  |  |
| 2.  |  |  |  |  |  |
| 3.  |  |  |  |  |  |
| 4.  |  |  |  |  |  |
| 5.  |  |  |  |  |  |

### Confusion matrix

<!-- ⚠️ TYPED AS A MARKDOWN TABLE. The notebook prints one ready to paste.
     An image of a matrix earns nothing. -->

| true \ predicted |  |  |  |
|---|---|---|---|
| **** |  |  |  |
| **** |  |  |  |
| **** |  |  |  |

**My biggest off-diagonal number, and what it means:**
<!-- Not "the model made mistakes" — WHICH boundary it didn't learn, and which
     direction. "7 real analysis posts were called hot_take and only 3 went the
     other way" is a direction, not just an error rate. -->



---

## Verdicts and Diagnoses

<!-- MET or MISSED against LAST UNIT's target. The target has to hold across
     all three seeds, not turn up sometimes. -->

| # | Criterion | Target | Verdict | How I decided |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |

**Diagnoses**

<!-- For each miss: the cause, and how you know. The four common causes are:
     too few examples for a label, a boundary you applied inconsistently, a
     genuinely hard label pair, and a task the model can't reach from this
     much data.

     ⚠️ Use your agreement report as evidence. It is the only instrument you
     have that can tell a LABELLING problem from a MODEL problem, and this
     section is graded on whether you used it that way. -->



---

## Agreement Report

<!-- Your rate against the staff set, and every disagreement adjudicated.

     Remember you labelled these 30 under the STAFF taxonomy in
     data/staff_taxonomy.md, not your own — so every argument below is made
     from those definitions and those decision rules. -->

**Agreement rate:** ___ / 30 = ___%

<!-- Nobody grades this number. A 60% who argues every disagreement from the
     stated rules beats a 95% who wrote "staff was right" nine times. Several
     of the 30 were chosen because they're genuinely ambiguous — you should be
     winning some of these. -->

**Disagreements**

<!-- Three lines each: the post, both labels, and who you think is right and
     why — grounded in the staff definitions you were both applying.

     Then sort each into one of three piles:
       (a) the rule covered it and I applied it loosely → a consistency problem
       (b) the rule genuinely doesn't say               → a gap in the definitions
       (c) the rule is ambiguous here and my reading is defensible → argue it.
           This is a legitimate win.

     Pile (a) is the one that matters most for your diagnosis: if you applied a
     written rule two different ways on 30 posts, that is direct evidence about
     what you did across your own 200. -->

**1.**
> *The post:*
>
> *Staff said / I said:*
>
> *My call, and why:*
>
> *Which pile:*

**2.**
> *The post:*
>
> *Staff said / I said:*
>
> *My call, and why:*
>
> *Which pile:*

**What the pattern in my disagreements tells me:**



---

## The Improvement

**What I changed:**

**Which diagnosis pointed at it:**

### Run Log — After

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1.  |  |  |  |  |  |
| 2.  |  |  |  |  |  |
| 3.  |  |  |  |  |  |
| 4.  |  |  |  |  |  |
| 5.  |  |  |  |  |  |

**Did it help, and how do I know:**

<!-- If it didn't, say so. Relabelling that didn't help is a genuinely
     interesting result and earns full credit. -->



---

## What's Still Broken

<!-- For each criterion still missed: what you'd do, and why you stopped. -->



**The gap between what I meant my labels to capture and what the model
learned:**
<!-- Two sentences. Your confusion matrix is the evidence. -->



<!-- ═════════════════════════════════════════════════════════════════════

     SUBMISSION CHECKLIST — unit 5

       [ ] criteria.md has five numbered criteria, each naming a NUMBER
       [ ] Each has a reason underneath tied to your data or taxonomy
       [ ] labels.csv: at least 150 rows, text/label/note, ONE file not split
       [ ] No label above 70%
       [ ] All five unit 5 sections have real content
       [ ] Label Taxonomy includes the decision rule for your hardest boundary
       [ ] The Dataset includes three hard cases
       [ ] results.json and test_split.csv committed (the notebook does this)
       [ ] At least four new commits
       [ ] Repository URL submitted — WRITE IT DOWN

     SUBMISSION CHECKLIST — unit 6

       [ ] Baseline vs. Trained table, with your prediction written beforehand
       [ ] Run Log — Before, five criteria across three seeds
       [ ] Confusion matrix TYPED AS A MARKDOWN TABLE
       [ ] A verdict on every criterion
       [ ] A diagnosis for every miss, using the agreement report as evidence
       [ ] Agreement Report with every disagreement adjudicated
       [ ] One improvement, with Run Log — After
       [ ] What's Still Broken
       [ ] results_three_seeds_before.json, results_three_seeds_after.json,
           baseline_results.json and
           agreement_results.json committed
       [ ] At least four new commits
       [ ] The SAME repository URL as last unit

     Do not delete and recreate this repository.
     ═════════════════════════════════════════════════════════════════════ -->

---

📖 **How to run this project: [RUNNING.md](RUNNING.md)**
