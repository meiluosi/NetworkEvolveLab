# Human validation codebook (v1)

You are judging **one model decision at a time**. You will not see which model
produced it. There is no single "correct" answer; use the codebook consistently.

## What you see

| column | meaning |
|---|---|
| `event_uid` | anonymous event id; do not edit |
| `listener_archetype` | `hub`, `bridger`, or `tail` |
| `listener_stance` | listener's scalar stance on the issue, in [-1, 1] |
| `listener_profile` | a verbal rendering of the listener's stance |
| `peer_message` | what the peer said to the listener |
| `tie_strength` | current directed tie weight, in [0, 1] |
| `model_action` | action chosen by the (hidden) model |

## The four actions

- **keep** — continue the current relationship; normal exposure continues.
- **mute** — keep the tie but reduce exposure / attention.
- **unfollow** — sever the directed tie; the listener no longer follows the peer.
- **block** — sever the tie and block the peer.

## Q1: Is the model action reasonable?

Choose exactly one:

- **reasonable** — a sensible agent in this state could plausibly take this action.
- **unreasonable** — the action contradicts the listener's state, the peer's
  message, or the tie's value in an obvious way.
- **cannot_judge** — the event is internally inconsistent or too ambiguous.

"Reasonable" does **not** mean "the action I would have chosen". A different but
defensible action is still reasonable.

## Q2: What would you have chosen?

Choose exactly one: `keep`, `mute`, `unfollow`, or `block`.
This is your preferred action, not a prediction of the model.

## Q3: Confidence

1 = very unsure, 3 = moderately sure, 5 = very sure.

## Examples

1. Listener stance +0.72, peer message strongly +0.68, tie 0.9:
   keep is reasonable; mute is defensible but conservative; unfollow is
   unreasonable because the peer is aligned and the tie is strong.
2. Listener stance -0.65, peer message strongly +0.70, tie 0.2:
   unfollow or mute are reasonable; keep is weak unless the listener is a
   tolerant bridger.
3. Listener stance 0.0, peer neutral, tie 0.8:
   keep is reasonable; unfollow is hard to justify without extra context.

## Rules

- Fill every row; do not leave Q1 or Q2 blank.
- Use only the exact tokens above (lowercase).
- Do not edit `event_uid`, stance, profile, message, tie, or model action.
- Do not search for the source model or the paper's hypotheses.
- Work independently; do not discuss individual rows with the other annotator
  until both sheets are submitted.
- If you recognise a row from a previous pass, still rate it fresh.

## Suggested pace

About 30–60 seconds per row; 300 rows ≈ 3–5 hours. Take breaks every 45–60
minutes to avoid drift. A 20-row pilot is recommended before the full set.
