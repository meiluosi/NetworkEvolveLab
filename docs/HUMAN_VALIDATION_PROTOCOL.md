# Human validation protocol (not executed)

Status: **protocol only; no human annotations were collected.** The paper must
not claim human validation. This file exists so a future revision can run the
check without redesigning it.

## Sampling

- Source: the 300 high-coverage receipts in `data/llm_archive/MiniMax-M3/`
  (one receipt per round; each contains the batch prompt and response text).
- Sample 300 applied decisions, stratified by:
  - LLM action (`keep` / `mute` / `unfollow` / `block`),
  - rule action on the same event (`reinforce` / `decay` / `block`),
  - coverage bucket (low-coverage archived arm vs 19.3% pilot).
- Two annotators per decision, blind to model identity.

## Question

"For the listener's stated belief and the speaker's message, is the chosen
relationship action reasonable under the codebook?" Options: reasonable /
unreasonable / cannot judge. Annotators also record whether they would have
chosen a different action.

## Agreement

- Cohen's kappa between annotators on the binary reasonable/unreasonable label.
- Fleiss' kappa if more than two annotators.
- Report disagreements and adjudication; report the LLM-vs-rule agreement from
  the paper as a separate, non-human baseline, not as human validation.

## Minimum bar for a top-venue claim

- >=300 decisions, >=2 annotators, kappa >= 0.6, and a written adjudication rule.
- If kappa < 0.6, report the protocol as inconclusive and do not use it to
  support or refute the structural null.

## Why it is not run

No annotators or annotation budget were available in this project. The paper
therefore reports the structural null with the per-event rule-agreement and
placebo controls only, and lists the absence of human validation as a limitation.
