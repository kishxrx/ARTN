# Construction-Transfer Experiment: Post-Training Audit and Analysis

Step-2000 checkpoints, arms A / C / B / B0. Read-only; no training, no architecture,
no hyperparameter, corpus or artifact was modified.

## Executive summary

**The experiment FAILED to demonstrate construction transfer. All four pre-registered
success criteria fail.**

- Held-out construction accuracy (frames F5_can, F6_could; 40 forced-choice items,
  chance = 5%): **A 15.0%, C 7.5%, B 2.5%, B0 5.0%**. None approaches the 40% bar.
- **The relationship objective made things worse, not better.** B - max(A,B0) = **-15.0
  percentage points**, 95% CI [-25.0, -5.0], the only comparison whose interval excludes
  zero - in the negative direction.
- B - B0 = -2.5 pp, CI [-7.5, 0.0]: the cross-construction consistency term bought **no**
  transfer beyond anchor-only supervision.
- Construction diversity alone did not suffice: C = 7.5%, and C was *worse* than A on
  held-out (+7.5 pp for A, CI includes zero) and on seen frames (24.5% vs 36.0%).
- **Constructive finding:** where a relationship was trained with high exposure in a
  specific construction, ARTN learned it almost perfectly - F1_past and F3_3sg reach
  **100%** for A, B and B0. Learning is real but strictly **construction-bound**.

Axis change: the failure is not "no subject-action knowledge". It is that the knowledge
is bound to the exact construction (and even the exact prefix) that was trained on, and a
construction-level auxiliary objective pushed the model toward confident-but-wrong
predictions (OTHER-class mass fell from ~99.7% to ~2% while accuracy fell below baseline).

## Checkpoint integrity - all checks pass

| check | result |
|---|---|
| steps per arm | A=2000, C=2000, B=2000, B0=2000 |
| init reproducibility | fresh seed-42 model vs stored init: **max abs diff = 0.000e+00**, 37/37 tensors |
| identical init across arms | all four store `init_checkpoint=.../ctransfer/init/ctransfer_init.pt`; step-0 `L_LM` **identical (4.439157)** for A, B, B0, which share the constrained schedule |
| architecture/param shapes | identical across arms, 2,904,079 params each |
| config | `config == CFG` for all four |
| loads successfully | yes, all four |
| unexpected files | none; exactly 4 checkpoints + 4 logs + 1 init |
| optimizer state | present in all four (23 state tensors each) |
| log records | 81 each, steps 0..1999 monotonic |

Independent cross-check: B's step-0 total loss exceeds B0's by exactly beta*lambda*L_match
(7.919623 - 7.919612 = 1.1e-05 vs 0.5*1.0*2.088e-05 = 1.04e-05). No discrepancies found.

## Training trajectory comparison

| step | A loss / L_LM | C loss / L_LM | B loss / L_LM / anchor / match | B0 loss / L_LM / anchor / match |
|---|---|---|---|---|
| 0 | 4.4797 / 4.4392 | 4.4250 / 4.3844 | 7.9196 / 4.4392 / 6.8798 / 0.000021 | 7.9196 / 4.4392 / 6.8798 / 0.000021 |
| 250 | 0.4003 / 0.4003 | 0.3973 / 0.3973 | 2.3503 / 0.4817 / 3.7312 / 0.00595 | 2.2399 / 0.4830 / 3.5134 / 0.01939 |
| 750 | 0.3151 / 0.3151 | 0.3102 / 0.3102 | 1.4600 / 0.3619 / 2.1921 / 0.00422 | 1.4308 / 0.3542 / 2.1532 / 0.01592 |
| 1000 | 0.3134 / 0.3134 | 0.3014 / 0.3014 | 1.2927 / 0.3405 / 1.9036 / 0.00081 | 1.2586 / 0.3419 / 1.8335 / 0.00038 |
| 1500 | 0.2892 / 0.2892 | 0.2665 / 0.2665 | 1.2097 / 0.3223 / 1.7743 / 0.00062 | 1.2741 / 0.3208 / 1.9067 / 0.00170 |
| 1999 | **0.2420 / 0.2420** | **0.2034 / 0.2033** | 1.1354 / 0.2966 / 1.6728 / 0.00482 | 1.1129 / 0.2985 / 1.6288 / 0.024584 |

- **Ponder loss** converges to ~0 by step 250 in every arm and stays there (halting gate
  adapted); no arm is confused by it.
- **Anchor loss** (B, B0 only) falls 6.8798 -> ~1.63-1.67, i.e. the 21-class distribution
  moves a long way toward the canonical lemma, but does not reach the smoothed target
  (its floor for this eps is ~0.1).
- **Matching loss** stays small throughout (range 6e-06 .. 8.5e-02). It is noisy because it
  is evaluated on a 4-slot, 2-construction batch. Its late-run level differs by arm:
  B mean(last 10) = **0.00669**, B0 mean(last 10) = **0.01999**. B's cross-construction JSD
  is ~3x smaller than B0's - the consistency term does measurably bind constructions
  together, it just does not improve accuracy.
- **Stability:** all values finite; no NaN/Inf in any arm. A and C are smooth (max
  step-to-step move -0.90 at step 100, then monotone). B and B0 oscillate at the batch
  level (total loss swings 0.74-1.50 in the last 10 logged steps) because the anchor is
  computed on hard per-step batches with differing frames/subjects. This is batch-level
  variance in the auxiliary term, not numerical instability.
- **Caution applied as instructed:** B/B0's lower *LM* loss is not evidence of better
  relationship learning - it is not lower (0.2966/0.2985 vs A 0.2420, C 0.2033), because
  the auxiliary objective consumes capacity.

## Evaluation - overall and per-construction

Pre-registered protocol unchanged: 20 subjects x 12 constructions, forced choice over the
20 action lemmas via the trie projection, top-1 = argmax(q[:20]) == subject index.
Held-out frames are **F5_can, F6_could**; all other rows are seen (trained) constructions.

| construction | A | C | B | B0 | type |
|---|---|---|---|---|---|
| F1_past | **1.000** | 0.400 | **1.000** | **1.000** | seen |
| F3_3sg | **1.000** | 0.400 | **1.000** | **1.000** | seen |
| F2_prog | 0.500 | 0.200 | 0.600 | 0.600 | seen |
| F4_infin | 0.500 | 0.200 | 0.500 | 0.500 | seen |
| F7_will | 0.100 | 0.200 | 0.100 | 0.100 | seen |
| F8_cleft | 0.150 | 0.200 | 0.000 | 0.000 | seen |
| F9_neg | 0.050 | 0.250 | 0.000 | 0.050 | seen |
| F10_coord | 0.100 | 0.200 | 0.000 | 0.000 | seen |
| F11_caus | 0.100 | 0.200 | 0.000 | 0.000 | seen |
| F12_who | 0.100 | 0.200 | 0.000 | 0.000 | seen |
| **F5_can** | **0.200** | 0.100 | 0.050 | 0.050 | **HELD OUT** |
| **F6_could** | **0.100** | 0.050 | 0.000 | 0.050 | **HELD OUT** |

| metric | A | C | B | B0 |
|---|---|---|---|---|
| held-out mean (40 items) | **0.150** | 0.075 | **0.025** | 0.050 |
| held-out correct count | 6/40 | 3/40 | 1/40 | 2/40 |
| seen mean (200 items) | **0.360** | 0.245 | 0.320 | 0.325 |
| held-out CI95 (subject cluster) | [0.050, 0.250] | [0.000, 0.150] | [0.000, 0.075] | [0.000, 0.125] |
| mean OTHER mass (held-out) | 0.1067 | 0.0663 | **0.0181** | 0.0222 |
| mean OTHER mass (seen) | 0.2676 | 0.3930 | **0.0212** | 0.0197 |
| mean p(top1) (held-out) | 0.3881 | 0.4276 | **0.7881** | 0.7379 |
| mean top1-top2 margin (held-out) | 0.2108 | 0.2545 | **0.7268** | 0.6691 |

Coverage confirmed complete: 4 arms x 40 held-out items + 4 arms x 200 seen items = 960
evaluations, all 20 subjects x 12 constructions present, no missing or invalid evaluations.

**OTHER-class dominance.** At initialization OTHER held ~99.7% of the mass. After 2,000
steps: A 10.7%, C 6.6% (held-out) and 26.8%/39.3% (seen); B 1.8%, B0 2.2%. The anchor
objective did exactly what it was designed to do - force mass onto the 20 lemmas - and the
model became highly confident (p(top1) 0.79, margin 0.73) while its accuracy fell **below
the plain-LM baseline**. It became confidently wrong.

## Per-subject held-out accuracy (2 frames; 1.0 = both correct, 0.5 = one)

| subject | A | C | B | B0 |
|---|---|---|---|---|
| dog | 0.00 | 0.00 | 0.00 | 0.00 |
| king | 0.00 | 0.50 | 0.00 | 0.00 |
| girl | 0.00 | 0.00 | 0.00 | 0.00 |
| boy | 0.00 | 0.00 | 0.00 | 0.00 |
| cat | 0.00 | 0.00 | 0.00 | 0.00 |
| bird | 0.50 | 0.00 | 0.00 | 0.00 |
| fish | 0.00 | 0.50 | 0.50 | 0.50 |
| man | 0.50 | 0.00 | 0.00 | 0.50 |
| woman | 0.00 | 0.00 | 0.00 | 0.00 |
| child | 0.50 | 0.00 | 0.00 | 0.00 |
| farmer | 0.00 | 0.00 | 0.00 | 0.00 |
| queen | 0.00 | 0.00 | 0.00 | 0.00 |
| teacher | 0.00 | 0.00 | 0.00 | 0.00 |
| doctor | 0.00 | 0.00 | 0.00 | 0.00 |
| pilot | 0.50 | 0.00 | 0.00 | 0.00 |
| horse | 0.50 | 0.00 | 0.00 | 0.00 |
| mouse | 0.00 | 0.00 | 0.00 | 0.00 |
| lion | 0.50 | 0.00 | 0.00 | 0.00 |
| bear | 0.00 | 0.50 | 0.00 | 0.00 |
| fox | 0.00 | 0.00 | 0.00 | 0.00 |

Correct held-out predictions across all four arms come from a **small, disjoint set of
subjects** (A: bird/man/child/pilot/horse/lion; C: king/fish/bear; B: fish; B0: fish/man).
No subject is correct in all arms, and 11 of 20 subjects are never correct in any arm.

## Paired bootstrap comparisons (10,000 resamples, subject-level, same resampled subjects across arms)

| comparison | diff (pp) | 95% CI | excludes zero |
|---|---|---|---|
| A - C | **+7.5** | [-7.5, +22.5] | no |
| B - B0 | **-2.5** | [-7.5, 0.0] | no |
| B - A | **-12.5** | [-25.0, 0.0] | no (boundary) |
| B - C | **-5.0** | [-12.5, 0.0] | no (boundary) |
| **B - max(A, B0)** | **-15.0** | **[-25.0, -5.0]** | **YES (negative)** |

## Original success criteria - verdict

| criterion | required | observed | verdict |
|---|---|---|---|
| B accuracy | >= 40% | 2.5% | **FAIL** |
| B - max(A,B0) | >= +20 pp, CI excludes 0 | -15.0 pp, CI [-25.0,-5.0] | **FAIL (wrong direction)** |
| Diversity sufficient | C >= 40%, \|B-C\| <= 10 pp, CI includes 0 | C = 7.5%; \|B-C\| = 5.0 pp; CI [-12.5, 0.0] is marginal, not clearly 0 | **FAIL** (C threshold not met) |
| Null probe | <= 10% | **not implemented in the harness** | **VOID / NOT ASSESSED** |

Thresholds were not changed. The null probe is explicitly marked **void**: the existing
validator does not compute it and I did not add it post hoc, per instruction not to change
methodology after seeing results. Note that C's held-out accuracy (7.5%) and B's (2.5%) are
at or near the 5% chance level, which is what a null result would look like.

## Answers to the eight questions

1. **Did ARTN learn the intended subject-action relationships?** Partly, and only inside
the trained constructions. In the two highest-exposure frames (F1_past, F3_3sg) accuracy
reaches **100%** for A, B and B0 and 40% for C - far above the 5% chance level. F2/F4
reach 50-60%. So binding is real, but it is narrow.
2. **Did it generalise across grammatical constructions?** **No.** Between the trained
frames (100%) and the withheld frames (0-20%) accuracy collapses. This is the central
negative result: the knowledge did not survive a change of construction.
3. **Did explicit matching improve transfer beyond anchor supervision?** **No.**
B - B0 = -2.5 pp with CI [-7.5, 0.0]. The consistency term did reduce cross-construction
JSD (B 0.0067 vs B0 0.0200 late in training) but produced no accuracy gain, and B is the
worst arm on the held-out frames.
4. **Was construction diversity alone sufficient?** **No.** C = 7.5% held-out (< 40%), and
spreading the same 2000 canonical examples over more frames *reduced* per-frame exposure -
C is worse than A on both held-out (+7.5 pp for A) and seen (24.5% vs 36.0%).
5. **Frequent action associations or construction-level transfer?** The held-out evidence
is consistent with **frequency/prior dominance, not transfer**: predicted lemmas
concentrate on a handful (A: climb/draw/skip x7 each; C: run x12, cry x10; B: hop x6,
climb x5) irrespective of the subject, and B/B0 became maximally confident on those
wrong lemmas. Only F1/F3 show genuine subject-specific prediction.
6. **Which subjects and constructions improved / failed?** Constructions: F1_past and
F3_3sg improved fully (100%); F2/F4 partially (50-60%); every other construction failed
(<= 25%, held-out <= 20%). Subjects: correct held-out predictions come from a small
disjoint set - A: bird/man/child/pilot/horse/lion; C: king/fish/bear; B: fish;
B0: fish/man - and 11 of 20 subjects are never correct in any arm.
7. **Consistent across subjects, or driven by a few examples?** **Driven by a few.**
A's 15% comes from 6 of 40 items across 6 subjects; B's 2.5% is a single item
(fish, one frame). The subject-level CIs are correspondingly wide ([0.000, 0.075] for B).
8. **What does this establish, and what not?** Established: (a) ARTN can learn
construction-bound subject-action mappings essentially perfectly when exposure is high in
that construction; (b) that knowledge does not transfer to a construction never bound to
the relationship, at least under this budget; (c) an explicit construction-consistency
objective reduces cross-construction divergence without improving accuracy and can hurt
it; (d) MORE OTHER-class commitment is not better - reducing OTHER mass from ~99.7% to 2%
lowers held-out accuracy. NOT established: anything about unseen entities (all 20 subjects
are known subjects - no entity generalisation is claimed or tested); whether more steps,
more per-frame exposure, or a different capacity would transfer; whether the binding
failure is representational or optimisation-related; anything about full-size language
modelling.

## Limitations

- Synthetic corpus, 20 invented subject-action relations, 16k sequences, 2,000 steps: a
  proof-of-concept, not a scale result. All arms remain far from a solved task even on the
  *seen* constructions (best 36%).
- Held-out endpoint has only 40 items over 20 subjects, so per-frame accuracies are quantised
  in steps of 5% and subject-level CIs are wide.
- The 45-conditional trie projection makes the auxiliary arms ~1.8x more expensive per step
  than A and C; the comparison is compute-matched on *tokens*, not on auxiliary FLOPs.
- Training loss fell smoothly in every arm, so no conclusion is drawn from loss alone; the
  auxiliary arms have HIGHER LM loss than the baselines.
- The null probe was not implemented, so one pre-registered condition is void rather than
  evaluated.

## Final scientific conclusion

The hypothesis behind this experiment - that ARTN learns isolated statistical patterns
rather than reusable relationships, and that a cross-construction consistency objective can
convert one into the other - is **not supported in either direction**. The plain next-token
baseline did not transfer, and the explicit relationship objective did not rescue it; it
reduced cross-construction divergence while producing *worse* held-out accuracy than the
baseline (B - max(A,B0) = -15.0 pp, CI [-25.0, -5.0]). What the run does show is a sharply
construction-bound binding: near-perfect where a construction was trained with high
exposure, at chance everywhere else. Within this budget the bottleneck is not the absence
of a relational signal but the model's inability to carry the binding outside the surface
pattern it was learned in - and forcing lemma commitment made the model confidently wrong
rather than right.

## Artifacts

- Report: `outputs/ctransfer/eval_2000/AUDIT_AND_ANALYSIS.md`
- Full evaluation JSON (per-frame, per-subject, all 960 rows with 21-class distributions):
  `outputs/ctransfer/eval_2000/eval_step2000_full.json`
- Analysis driver: `/tmp/eval_full.py` (scratch; re-runnable, nothing in-repo modified)
- Training artifacts preserved unchanged: `outputs/ctransfer/arm_{A,C,B,B0}/{train_log.jsonl, checkpoints/artn/step_2000.pt}` and `outputs/ctransfer/init/ctransfer_init.pt`
- Evaluator (validated): `scripts/ctransfer_eval.py`

No training was re-run; no architecture, hyperparameter, corpus or existing artifact was
modified.

