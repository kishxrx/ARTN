# ARTN Failure Analysis Log

> Stage 11: Documenting what broke, why it failed, and how we engineered the fixes.

---

## Purpose

This document preserves hard-won lessons from the ARTN project. Every failure is valuable intelligence that should improve future judgment.

**CONSTITUTIONAL COMPLIANCE:** Per Article IV, truth and correctness take precedence over convenience. We document failures honestly to prevent repeated mistakes (Article VI).

---

## Template for Logging Failures

```markdown
### Failure ID: F-XXX

**Date:** YYYY-MM-DD
**Stage:** Which experiment stage
**Symptom:** What broke / behavior observed
**Root Cause:** Why it happened
**Investigation Steps:** How we diagnosed it
**Fix:** What we changed
**Verification:** How we confirmed the fix worked
**Lesson Learned:** What we know now that we didn't before
**Related Files:** Links to relevant code/configs
```

---

## Failures

*No failures logged yet. This section will be populated as we progress through stages.*

---

## Architectural Design Decisions

This section documents key decisions made during development and their rationale.

### ADR-001: Two-Era Decoupled Roadmap
**Status:** Accepted
**Context:** Training on chemical structures (SMILES) has higher risk and complexity.
**Decision:** Complete Era 1 (TinyStories) first to secure a portfolio piece before attempting Era 2.
**Consequences:** Lower project risk, validated architecture before domain specialization.

### ADR-002: Stage-by-Stage Ablation
**Status:** Accepted
**Context:** Multiple architectural innovations combined could fail unpredictably.
**Decision:** Isolate each optimization in its own experiment stage with metrics logging.
**Consequences:** Clear attribution of performance/quality changes to specific changes.

### ADR-003: Adafactor over AdamW
**Status:** Accepted
**Context:** 4GB RAM constraint makes Adam's per-parameter momentum expensive.
**Decision:** Use Adafactor which factorizes momentum into row/column sums.
**Consequences:** ~60% optimizer memory reduction, slightly different training dynamics.

---

## Performance Observations

*To be populated with benchmark data.*

---

## References

- PIOS Constitution Article VI: "PIOS is considered architecturally compromised if it... Repeats previously learned mistakes."
