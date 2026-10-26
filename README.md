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

**Base model:** `distilbert-base-uncased`

**Settings:** 3 epochs, learning rate 2e-5, batch size 16, max_length 128, seed 42

**Training device:** `cuda` (`NVIDIA GeForce RTX 2070`) using PyTorch `2.14.1+cu130`. The entire three-epoch training run took 13.9 seconds.

**Anything I changed from the defaults, and why:**
I kept the starter defaults (3 epochs, learning rate 2e-5, batch size 16, and max_length 128). In an online sports forum like `r/nba`, 128 tokens comfortably covers almost every post without unnecessary padding overhead, and 3 epochs with AdamW at 2e-5 learning rate provides smooth convergence without over-fitting the 146-example training split.

**Split sizes:**
- Train: 146 posts (`analysis`: 49, `hot_take`: 48, `reaction`: 49)
- Val: 32 posts (`analysis`: 11, `hot_take`: 11, `reaction`: 10)
- Test: 32 posts (`analysis`: 10, `hot_take`: 11, `reaction`: 11)

No label had fewer than 8 examples in the test split (all labels have 10 or 11 examples), ensuring that held-out evaluation metrics remain robust against small-sample score volatility.

---

## How I Used AI

**Moment 1**

- *What I asked for:* I provided my initial label definitions and asked an AI assistant to generate challenging edge-case posts situated directly on the boundary between `analysis` and `hot_take`, specifically focusing on heated rants that cited isolated statistics or player salaries.
- *What came back:* The assistant produced multiple plausible posts where posters cited single numbers (e.g., "He scored 38 points on 14 shots but is a defensive traffic cone" or "making $45M to shoot 30%").
- *What I changed:* I realized that a keyword heuristic for numbers was insufficient. I formulated and formalized Decision Rule 1: a checkable fact must be load-bearing and carry the argument. If a stat is cited purely as rhetorical decoration or an afterthought to an unsupported claim, it is classified as `hot_take`.

**Moment 2**

- *What I asked for:* I shared my proposed acceptance criteria targets and asked for a critical evaluation of potential failure modes and split sensitivity given our sample size.
- *What came back:* The AI noted that with 210 total posts, the 15% test split contains only ~10-11 examples per label, meaning a single misclassification swings a per-class F1 score by 5 to 8 percentage points. It warned that setting targets above 0.75 would make the criteria brittle across different random seeds in Unit 6.
- *What I changed:* I adjusted my per-class targets to defensible thresholds (0.70 for `analysis`, 0.65 for `reaction`, and 0.60 for `hot_take`) and explicitly documented sample-size sensitivity in `criteria.md`.

**Pre-labelling disclosure:**
The first 20 posts in `labels.csv` were labeled entirely by hand without any AI tools to establish an honest baseline (flagged as `cold` in the `note` column). For the remaining 190 rows, an AI tool was used to produce initial pre-label suggestions from our taxonomy prompt. I then personally read, audited, and verified every single row, correcting misclassified edge cases in accordance with our decision rules. All pre-labeled rows are transparently marked as `pre-labeled, verified` in `labels.csv`.

<!-- ═══════════════════════ UNIT 6 — THE TEST ═══════════════════════

     Don't fill these in during unit 5.
     ═══════════════════════════════════════════════════════════════════ -->

---

## Baseline vs. Trained

<!-- Both models on the same posts. `python baseline.py --trained results.json`
     prints this table for you. -->

| Measure | Baseline | Trained | Difference |
|---|---|---|---|
| Overall accuracy | 0.531 | 0.938 | +0.406 |
| Macro F1 | 0.476 | 0.936 | +0.460 |
| F1 — `analysis` | 0.563 | 0.900 | +0.337 |
| F1 — `hot_take` | 0.700 | 0.909 | +0.209 |
| F1 — `reaction` | 0.167 | 1.000 | +0.833 |

**What I predicted before I looked:**
I predicted that the fine-tuned model would substantially beat the zero-shot baseline on `reaction` and `hot_take`, because sports forum slang, abbreviations, sarcasm, and emotional exclamations require specialized domain grounding that generalist NLI hypotheses struggle to distinguish. I also anticipated the fine-tuned model would achieve markedly higher precision on `analysis` by recognizing load-bearing basketball statistics rather than being fooled by incidental or decorative numerical mentions.

**What the gap actually means:**
The overall accuracy gap (+0.406, improving from 53.1% to 93.8%) and Macro F1 improvement (+0.460, from 0.476 to 0.936) confirms that domain-specific fine-tuning added crucial discrimination that zero-shot generalist models lack without training. In particular, the baseline model achieved an abysmal 0.167 F1 on `reaction` (recall of only 9.1%), repeatedly mistaking emotional fan outpourings and hyperbole (such as *"That fourth quarter took years off my life. I need a week of uninterrupted sleep to recover"*) for analytical commentary. Fine-tuning taught DistilBERT to separate structural arguments from visceral fan reactions.

Furthermore, running the baseline once with bare label names rather than full definitions yielded an `analysis` F1 of 0.000 (0 out of 10 identified), demonstrating that our written definitions provided essential semantic grounding, but local fine-tuning was necessary to resolve the actual community discourse boundaries. Finally, the zero-shot baseline took ~45 seconds on GPU and several minutes on CPU across just 32 samples, whereas our fine-tuned DistilBERT runs inference in under 10 milliseconds per sample with zero external dependencies.



---

## Run Log — Before

<!-- Five criteria across three seeds. The notebook's section 6 prints the
     spread table; the Target and Verdict columns are yours. -->

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1. Overall accuracy | ≥ 0.68 | 0.969 | 1.000 | 1.000 | MET |
| 2. `analysis` F1 | ≥ 0.70 | 0.947 | 1.000 | 1.000 | MET |
| 3. `reaction` F1 | ≥ 0.65 | 1.000 | 1.000 | 1.000 | MET |
| 4. `hot_take` F1 | ≥ 0.60 | 0.957 | 1.000 | 1.000 | MET |
| 5. Confidence calibration spread | ≥ 0.15 | 0.100 | 0.000 | 0.000 | MISSED |

*Hardware Device:* `cuda` (NVIDIA GeForce RTX 2070, PyTorch 2.14.1+cu130).  
*Stability Note:* Overall accuracy moved 0.031 across seeds (from 0.969 to 1.000). A spread well under 0.05 indicates our dataset split and sample distributions are stable and consistent across random seeds.

### Confusion matrix

<!-- ⚠️ TYPED AS A MARKDOWN TABLE. The notebook prints one ready to paste.
     An image of a matrix earns nothing. -->

| true \ predicted | analysis | hot_take | reaction |
|---|---|---|---|
| **analysis** | 9 | 1 | 0 |
| **hot_take** | 0 | 11 | 0 |
| **reaction** | 0 | 0 | 11 |

**My biggest off-diagonal number, and what it means:**
My biggest off-diagonal number is 1, where 1 real `analysis` post was predicted as `hot_take` (and 0 in the reverse direction). The misclassified post (*"Their bench was outscored 48-18 tonight, marking the seventh consecutive game where the second unit posted a negative scoring differential"*) featured a strongly frustrated, evaluative tone that caused the model to lean toward classifying it as a venting take, despite the presence of checkable margin and game-streak facts. The model never confused `reaction` with any other class (11/11 correct).



---

## Verdicts and Diagnoses

<!-- MET or MISSED against LAST UNIT's target. The target has to hold across
     all three seeds, not turn up sometimes. -->

| # | Criterion | Target | Verdict | How I decided |
|---|---|---|---|---|
| 1 | Overall accuracy | ≥ 0.68 | MET | Achieved 0.969 on Seed 42, 1.000 on Seed 7, and 1.000 on Seed 2024. All three trials held well above the 0.68 threshold. |
| 2 | `analysis` F1 | ≥ 0.70 | MET | Achieved 0.947 on Seed 42, 1.000 on Seed 7, and 1.000 on Seed 2024. Maintained high precision and recall on tactical posts across all splits. |
| 3 | `reaction` F1 | ≥ 0.65 | MET | Scored a perfect 1.000 on Seed 42, 1.000 on Seed 7, and 1.000 on Seed 2024 (11/11, 10/10, 11/11 held-out items classified without error). |
| 4 | `hot_take` F1 | ≥ 0.60 | MET | Achieved 0.957 on Seed 42, 1.000 on Seed 7, and 1.000 on Seed 2024, easily surpassing the 0.60 target across all trials. |
| 5 | Confidence calibration spread | ≥ 0.15 | MISSED | Target required the most-confident third to beat the least-confident third by ≥ 0.15. Across seeds, spread was 0.100 on Seed 42 (1.000 vs 0.900), 0.000 on Seed 7 (1.000 vs 1.000), and 0.000 on Seed 2024 (1.000 vs 1.000). The target failed to hold across trials. |

**Diagnoses**

- **Why Criterion 5 Missed (Calibration Compression):**
  Criterion 5 missed not because the model was inaccurate, but because of a ceiling effect that compressed the calibration spread. On Seeds 7 and 2024, the model achieved 100% test accuracy across all 32 held-out posts. As a consequence, both the most-confident third and the least-confident third scored 1.000, mathematically reducing the spread to 0.000. On Seed 42, where 1 error occurred, the least-confident third still achieved 90.0% accuracy (against 100% in the top third), leaving a spread of 0.100, which fell short of our pre-registered 0.15 goal.
  
- **Connection to Agreement Check & Error Analysis:**
  Our agreement check against staff revealed that 3 of our 6 disagreements occurred on posts citing numbers, where superficial metrics were mistaken for load-bearing arguments. On Seed 42 of the trained model, the single error was a real `analysis` post (*"Their bench was outscored 48-18 tonight..."*) called `hot_take`. The post combined genuine metrics with an emotional complaint about bench depth, showing that while our hand-labeling distinguished facts from feelings, the model's confidence distribution saturates heavily toward 1.0 on short forum texts.



---

## Agreement Report

<!-- Your rate against the staff set, and every disagreement adjudicated.

     Remember you labelled these 30 under the STAFF taxonomy in
     data/staff_taxonomy.md, not your own — so every argument below is made
     from those definitions and those decision rules. -->

**Agreement rate:** 24 / 30 = 80.0%

<!-- Nobody grades this number. A 60% who argues every disagreement from the
     stated rules beats a 95% who wrote "staff was right" nine times. Several
     of the 30 were chosen because they're genuinely ambiguous — you should be
     winning some of these. -->

**Disagreements**

**1.**
> *The post:* "Carbon plates aren't the whole story. My times dropped 2% in the shoes and another 2% the following year in the same pair, when the only thing that changed was going from 35 to 50 miles a week."
>
> *Staff said / I said:* `hot_take` / `analysis`
>
> *My call, and why:* I believe **my reading is right**. While the opening sentence sounds contrarian, the rest of the post presents a controlled individual experiment with checkable numbers (2% drop in shoes, another 2% with volume increase from 35 to 50 miles/week). The facts do real work to carry the specific argument that weekly mileage explained the second 2% rather than shoe technology. Staff treated it as an unargued personal opinion, but under the written definitions, the fact carries the comparative argument.
>
> *Which pile:* **(c) the rule is ambiguous here and my reading is defensible.**

**2.**
> *The post:* "Three of us ran the same session on the same day, different shoes, within 10 seconds of each other over 5 miles. I'm not saying shoes do nothing, but whatever they do was smaller than the noise between three people."
>
> *Staff said / I said:* `hot_take` / `analysis`
>
> *My call, and why:* I believe **my reading is right**. Staff labeled this as `hot_take` presumably because comparing three people on one day is an anecdotal sample rather than a rigorous study. However, Rule 1 requires that the argument be backed by a specific, checkable fact where the fact is load-bearing. The post provides specific test parameters (three runners, same session, different shoes, 10s spread over 5 miles) and uses that observation directly to argue that shoe variance was smaller than interpersonal noise. The numbers do load-bearing work.
>
> *Which pile:* **(c) the rule is ambiguous here and my reading is defensible.**

**3.**
> *The post:* "Times are down about 3% since the shoe rules changed, so obviously the shoes are the whole story and the sport has a problem."
>
> *Staff said / I said:* `hot_take` / `analysis`
>
> *My call, and why:* **Staff is right.** I saw the checkable statistic ("down about 3%") and immediately flagged it as analysis. However, Rule 1 explicitly states: *"If the number is decoration on a conclusion the writer had already reached, it is a hot_take with a statistic in it."* The conclusion ("obviously the shoes are the whole story and the sport has a problem") is a sweeping, sarcastic generalization, and the 3% stat is merely decorative framing rather than a structured demonstration of causality. I applied the rule loosely because I saw a number.
>
> *Which pile:* **(a) the rule covered it and I applied it loosely.**

**4.**
> *The post:* "First run back after eight weeks out. Two miles, slower than my warm-up used to be, and I don't care even slightly."
>
> *Staff said / I said:* `reaction` / `analysis`
>
> *My call, and why:* **Staff is right.** The post contains checkable numbers ("eight weeks", "Two miles"), but those numbers do not build an analytical argument. Rule 2 clearly dictates that `reaction` reports what just happened to the writer and how it felt, whereas `analysis` requires an argument. The heart of the post is the emotional relief ("and I don't care even slightly"). I let numerical tokens override the fundamental purpose of the post.
>
> *Which pile:* **(a) the rule covered it and I applied it loosely.**

**5.**
> *The post:* "Wind was so bad on the out section I was leaning at about 20 degrees. Coming back felt like cheating."
>
> *Staff said / I said:* `reaction` / `hot_take`
>
> *My call, and why:* The disagreement stems from an unaddressed boundary in the taxonomy. The statement *"Coming back felt like cheating"* reads structurally like a humorous hyperbolic claim, while *"leaning at about 20 degrees"* is a subjective descriptive flourish. Staff classified it as `reaction` because it describes the writer's immediate physical experience during a run. The staff rules do not explicitly specify how to handle hyperbolic physical metaphors.
>
> *Which pile:* **(b) the rule genuinely doesn't say.**

**6.**
> *The post:* "Just realised the shoes I set every PB in this year are the ones I bought by accident in the wrong size."
>
> *Staff said / I said:* `hot_take` / `reaction`
>
> *My call, and why:* I believe **my reading is right**. Staff interpreted this as an implicit hot take asserting that shoe sizing is unimportant for performance. But linguistically, the post begins with *"Just realised"*—it captures an immediate personal revelation and humorous reflection on the writer's own footwear. Under Rule 2, a post reporting what just happened to the writer and how they experienced it is squarely a `reaction`.
>
> *Which pile:* **(c) the rule is ambiguous here and my reading is defensible.**

**What the pattern in my disagreements tells me:**
Half of my disagreements (Posts 1, 2, and 6) belong to **Pile (c)**, where I defended a principled reading against staff interpretation. However, the two disagreements in **Pile (a)** (Posts 3 and 4) uncover an important vulnerability in my annotation intuition: **I tend to over-index on the mere presence of numbers.** Whenever a post mentions percentages, weeks, or distances, I am tempted to label it `analysis`, even when the number is purely decorative framing for a hot take (`sp_014`) or emotional background for a reaction (`sp_023`). Recognizing this tendency explains why boundary rules like the Load-Bearing Fact Test are vital.

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

       [x] criteria.md has five numbered criteria, each naming a NUMBER
       [x] Each has a reason underneath tied to your data or taxonomy
       [x] labels.csv: at least 150 rows, text/label/note, ONE file not split
       [x] No label above 70%
       [x] All five unit 5 sections have real content
       [x] Label Taxonomy includes the decision rule for your hardest boundary
       [x] The Dataset includes three hard cases
       [x] results.json and test_split.csv committed (the notebook does this)
       [x] At least four new commits
       [x] Repository URL submitted — WRITE IT DOWN

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
