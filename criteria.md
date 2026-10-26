# Acceptance criteria — TakeMeter

Five criteria that say what "working" means for this classifier, written in
unit 5 **before** anything was trained.

**All five are yours this time.** None are given. You've had two projects of
practice.

An acceptance criterion names a number. *"The model is accurate"* is an
opinion. *"Every label has an F1 of at least 0.60 on the held-out set"* is a
criterion.

Under each, write a sentence or two on **why that number**. A reason that says
something about your data or your taxonomy earns credit — *"I picked 0.60 F1
for `reaction` because it's my smallest label and I only have about 50
examples of it"*. A reason that could be attached to any project does not.

> Missing your own targets next unit costs you nothing. Setting a target so
> easy you can't miss it does.

---

## Pick numbers you can defend

Not numbers that sound impressive. Three labels means a coin-flip guesser gets
about 33%, so a target of 0.40 is barely a target. Your number should sit
somewhere you'd honestly call useful.

**Cover at least three of these five areas.** They're here as prompts, not as a
form to fill in — a criterion that fits none of them is fine if it names a
number.

| Area | A question it could answer |
|---|---|
| Overall accuracy | How often does it need to be right to be worth using? |
| Per-label performance | Is one label allowed to be much worse than the others? |
| Balance | How lopsided can your label counts get before it's a problem? |
| Consistency | If someone else labelled the same posts, how often should you agree? |
| Confidence | Should a confident prediction be right more often than an unsure one? |

Two things worth knowing before you pick numbers, because both will affect
whether you hit them:

- **Your smallest label will have the jumpiest score.** If a label has 50
  examples, about 8 land in the test split. An F1 computed on 8 examples moves
  a lot between seeds. A target for that label should be looser than one for
  your biggest label, and saying so is a good reason.
- **Unit 6 tests across three seeds, and the target has to hold across all
  three.** A target of 0.65 against results of 0.71, 0.62, 0.68 is a **miss**.
  Pick with that in mind — it is stricter than it first sounds.

---

## 1. Overall accuracy reaches at least 0.68 on the held-out test split

The model achieves an overall classification accuracy of at least 0.68 on the held-out test set.

**Why this target:**
In a balanced three-class classification problem, a coin-flip or random guesser achieves approximately 0.33 accuracy, while a zero-shot model without domain fine-tuning typically scores around 0.45 to 0.55 on informal sports discourse. An accuracy of 0.68 requires the model to correctly categorize more than two out of every three posts on unseen data, confirming that fine-tuning captured meaningful distinctions without setting an unrealistically brittle floor across random seeds.

---

## 2. The analysis label achieves an F1 score of at least 0.70 on the held-out set

The F1 score for `analysis` on the held-out test set is at least 0.70.

**Why this target:**
Posts classified as `analysis` have the most distinctive and objective lexical markers in our taxonomy—namely concrete numbers, percentages, dates, dollar amounts, and structured comparative syntax (e.g., "per possession", "effective FG%", "compared to"). Because DistilBERT is particularly effective at picking up these explicit quantitative tokens, `analysis` should achieve our highest per-class performance target.

---

## 3. The reaction label achieves an F1 score of at least 0.65 on the held-out set

The F1 score for `reaction` on the held-out test set is at least 0.65.

**Why this target:**
`reaction` posts feature distinct emotional vocabulary, first-person pronouns ("I", "my"), and punctuation patterns (exclamation marks, all-caps). With roughly 70 total examples of `reaction` in our 210-post corpus, approximately 10 to 11 examples fall into the 15% held-out test split. Because a single misclassification on 10 examples swings the F1 score by roughly 0.05 to 0.08, a target of 0.65 demands strong discrimination while leaving appropriate headroom for small-sample variance across seeds.

---

## 4. The hot_take label achieves an F1 score of at least 0.60 on the held-out set

The F1 score for `hot_take` on the held-out test set is at least 0.60.

**Why this target:**
`hot_take` is our most linguistically varied category and sits directly on the taxonomy's hardest boundary (`analysis` vs. `hot_take`, where posts may contain ornamental stats or sweeping declarative judgements). Because it borders both emotional venting and assertive arguments, it is naturally the most vulnerable to boundary confusion. Setting the target at 0.60 ensures the model learns real discriminatory features for speculative claims without penalizing expected boundary difficulty.

---

## 5. Confidence calibration demonstrates an accuracy spread of at least 0.15 between the top and bottom confidence tiers

The classification accuracy on the most-confident third of held-out test predictions is at least 0.15 higher than the accuracy on the least-confident third (`accuracy_most_confident_third - accuracy_least_confident_third >= 0.15`).

**Why this target:**
A reliable classifier should be aware of its own uncertainty. The notebook's `confidence_report` splits test predictions into top and bottom thirds by softmax confidence score. If the model is genuinely calibrated, predictions on which it is confident should be significantly more accurate than predictions where its output distribution is near uniform. A 15 percentage-point gap verifies that the model's confidence scores provide genuine diagnostic value for human reviewers.



---

<!-- ─────────────────────────────────────────────────────────────────────────
     UNIT 6 — read this before you change anything above.

     If a criterion turns out to be BROKEN rather than merely unmet, you can
     revise it, and that earns credit. But never delete or edit the original
     line. Add the revision underneath, like this:

         ## 2. Every label performs acceptably

         The model performs well on all labels.

         **Why this target:** ...

         > **Revised in unit 6:** Every label has an F1 of at least 0.60 on
         > the held-out set.
         >
         > **Why revised:** "performs well" gave me nothing to check. I
         > couldn't produce a verdict from it at all.

     That's a revision because the criterion couldn't be MEASURED.

     Lowering a target because you missed it is not a revision, and it costs
     you the point:

         ✗ "Overall accuracy of at least 0.65" → "at least 0.55", because
            0.65 turned out to be optimistic for 200 examples.

     A number you missed stays where it is, gets diagnosed, and gets a fix
     attempted. That's where the points are.
     ───────────────────────────────────────────────────────────────────────── -->
