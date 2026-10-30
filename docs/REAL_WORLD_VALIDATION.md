# Real-world validation protocol

Three Review-3 concerns can only be settled with real people:

| Concern | What the repo can already show | What still needs real data |
|---|---|---|
| The mistake classifier is trained on synthetic solutions | Real student errors from a tutoring system (CogTutorBugs, typed): `validation_cogtutor.json` | Real *handwritten* student work, labelled by us |
| Phone photos and diverse handwriting | Simulated capture conditions and MNIST writer populations: `validation_propagation.json`, `validation_writers.json` | Real phone photos of real students' work |
| Adaptive practice is only validated in simulation | Robustness across five simulators: `validation_policy.json` | A classroom study |

Study A covers the first two rows; Study B the third. Both can run in the same classes: the
pre- and post-test sheets of Study B are the photographed work of Study A.

## Study A: photographed student work

**Goal.** Measure reading, diagnosis and the confirm-before-diagnose flow on real photos of real
students' working, and check how well the simulations predicted them.

**Participants and material.** At least 30 students (grades 6-9) from at least two classes. Each
writes a 12-item worksheet drawn from the item bank (1-2 items per skill S1-S8) on their own paper
with their own pen or pencil, showing their working. Target: **200 pages** (the 95% confidence
interval on an accuracy near 50% is then about ±7 points; at 100 pages it is ±10).

**Capture.** Each page is photographed with a phone in normal classroom conditions. A subset of at
least 40 pages is photographed twice more: with a second (cheaper) phone and under different
lighting, so capture effects can be measured on the same handwriting. Record `device`, `paper`
(plain / ruled / grid), `lighting` (daylight / indoor / shadow) and the writer ID.

**Annotation.** Two annotators label every page independently, without seeing StepWise's output:

1. `reference_text`: the true transcription in StepWise notation, one line per written line,
   `*` for a multiplication cross, `/` for division. Spacing does not matter.
2. `mistake_label`: the category of the *first* wrong step, using the rules below.

Disagreements are discussed and the agreed label kept in `mistake_label`; the second annotator's
original label goes in `mistake_label_b`, so the evaluation reports Cohen's kappa. Labels are
trusted only if kappa is at least 0.70.

| Category | Rule (first wrong step) | Example |
|---|---|---|
| `correct_solution` | every step valid and the answer matches the key | `3x - 6 = 9`, `3x = 15`, `x = 5` |
| `arithmetic_error` | right operation and signs, wrong value | `12 - 4 = 7` |
| `sign_error` | right size, wrong sign; a negative dropped or flipped | `-3 + 5 = -8`, `-2 * -3 = -6` |
| `incorrect_operation` | a different operation than the one written or needed | `12 - 4 = 16` |
| `algebraic_manipulation_error` | an invalid algebra move | `2x + 4 = 10` -> `2x = 14` |
| `skipped_step` | the answer is what omitting one operation gives, with no working shown | `2x + 4 = 10` -> `x = 6` |
| `final_answer_only_error` | all steps valid, final answer differs from the key | answer copied wrongly |
| `transcription_uncertainty` | the annotators cannot read it either | |
| `insufficient_information` | no working, or not mathematics | `idk` |

If two categories fit, prefer the more specific: sign > operation > arithmetic.

**Manifest and analysis.** One JSONL line per page, as in
[`evaluation/manifests/real_world_template.jsonl`](../evaluation/manifests/real_world_template.jsonl).
Keep images and the filled manifest in `evaluation/manifests/real_world/` (git-ignored). Then:

```bash
cd evaluation && python real_world_eval.py --manifest manifests/real_world/real_world.jsonl
```

It reports:

- reading (exact match, CER) overall and by device, paper, lighting, writer and grade;
- diagnosis from the true transcription, which is the mistake classifier alone and the real-work
  counterpart of the synthetic 0.992 macro-F1, against the rule-based baseline;
- diagnosis end to end from the photo, without and with confirm-before-diagnose, plus the number
  of symbols a student is asked to check per page;
- inter-annotator agreement.

All headline numbers come with bootstrap 95% confidence intervals. Compare them with the
simulated values in `backend/ml_models/reports/validation_propagation.json`.

**Ethics and data handling.** Permission from the school, informed consent from parents and
assent from students; students may withdraw at any time. No names on pages (writer IDs only);
crop out anything identifying before saving. Images stay on the team's machines, are never
committed or uploaded, and are deleted at the end of the project.

## Study B: classroom study of adaptive practice

**Goal.** Test whether StepWise's adaptive practice improves learning more than the same amount
of random practice from the same item bank.

**Design.** Two-arm randomized controlled trial with students as the unit, stratified by class
and pre-test tercile. The control gets random practice with the same item bank, the same number
of items and the same interface, so only item selection differs.

**Procedure.**

1. Pre-test: 16 items, 2 per skill (form A), no feedback.
2. Four practice sessions of 10 items each (40 items, as in the simulations).
3. Immediate post-test: form B. Forms are parallel: same templates, different numbers.
4. Delayed post-test after two weeks: form C. The simulations show forgetting erodes the adaptive
   advantage (`validation_policy.json`), so retention is measured directly.

**Outcomes.** Primary: immediate post-test score, analysed by ANCOVA with the pre-test as
covariate and class as a fixed effect. Secondary: delayed post-test; the number of skills moved
from not mastered to mastered; practice spent on skills already mastered at pre-test; time on task.

**Sample size** (two-sided alpha 0.05, power 0.80; ANCOVA with pre/post correlation 0.5):

| Effect size d | Students per arm |
|---|---|
| 0.3 | 132 |
| 0.4 | 75 |
| 0.5 | 49 |

The simulations (`validation_policy.json`) give d = 0.37 with the original simulator and d = 0.50 with
learners calibrated on real students, which needs 89 and 50 students per arm. With forgetting and
varied learning rates combined, though, d = 0.04, which no classroom study could detect. Simulated
effects are also optimistic, because a simulator cannot capture motivation, novelty or teacher
effects. A
first study with two classes (about 2 x 30 students) can only detect a large effect (d of about
0.7). Run it as a **pilot**: estimate the effect size, check feasibility (completion, logging,
time), and use the estimate to size a confirmatory study.

**Software needed before the study.** A per-student arm (`adaptive` or `random`) used by
`/api/practice/{id}/next`. The simulator already implements both policies in
`app/ml/irt/recommender.py::policy_choose`. The study also needs the arm logged with each practice
attempt, and test forms that give no feedback.

**Threats.** Students in different arms comparing notes (spillover), novelty effects, and teacher
effects. Stratify by class, have the same teacher run both arms, and keep the interface identical.

**Ethics.** As in Study A. Additionally, after the study the control group gets access to adaptive
practice.
