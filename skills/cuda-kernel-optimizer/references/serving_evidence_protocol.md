# Serving Evidence Protocol

Use the narrowest claim supported by the evidence. A faster instruction path is
useful evidence, but it is not a serving result.

## Claim ladder

| Layer | Minimum evidence | Allowed claim |
|---|---|---|
| Generated code | Compiler output or SASS bound to the candidate binary and target architecture | The intended mechanism was emitted for that binary. |
| Isolated operator | Reference validation and paired A/B timing on identical inputs and device conditions | The tested operator improved for that input set. |
| Matched runtime | Paired A/B runs with the same engine, model, inputs, batching, cache policy, and runtime settings | The implementation improved in that runtime configuration. |
| Serving endpoint | A clean window load test with identical model, request mix, concurrency policy, and endpoint configuration | The tested endpoint metric improved in that window. |

Generated code only proves that a mechanism was emitted; it does not prove that
the path executed or improved performance. Operator timing does not prove a serving benefit.
Generated-code or operator evidence alone cannot support a deployment claim.

## Connect the result to the run verdict

The outer evaluation belongs to the user-provided workload. It must preserve
the production objective and representative request distribution; a synthetic
kernel benchmark cannot replace it.

- `kernel_only_win` means the paired kernel result improved but the real
  workload did not confirm the configured end-to-end objective. Do not promote
  it as a serving win.
- `end_to_end_win` means the candidate passed the run's correctness and paired
  gates and the user-provided workload confirmed the configured objective. The
  claim is still limited to the captured workload, environment, and test
  window.

## Serving A/B evidence

Collect baseline and candidate measurements as paired A/B observations. Keep
model weights, request replay, admission control, batching, cache state,
warm-up, concurrency, and measurement boundaries identical. Randomize or
interleave pair order when time drift could bias one side.

A serving result also needs:

| Evidence | Record |
|---|---|
| Raw request evidence | Request timestamps or pair IDs, input-shape and sequence-length distributions, concurrency, response status, and the raw objective samples. Redact payload content when required, but retain stable request identities. |
| Environment evidence | GPU and host identity, software and model versions, clocks or power policy, container image, runtime flags, cache policy, and competing processes. |
| Clean window | Warm-up boundary, start and end time, stable traffic policy, error rate, and confirmation that maintenance or unrelated rollouts did not overlap. |
| Contamination check | GPU occupancy and host load before and during the run, plus scheduler or process evidence sufficient to detect shared-host contamination. |

If shared-host contamination cannot be ruled out, mark the serving comparison
inconclusive or repeat it in a clean window. Do not average away interference.
Retain raw request and environment evidence with the summary so another person
can reproduce the pairing and check the claim.

## Decision checklist

- [ ] The claim stops at the highest completed layer in the claim ladder.
- [ ] Correctness passed before any performance claim.
- [ ] The comparison used paired A/B samples with identical policies.
- [ ] The user-provided workload matches the intended serving objective and
      request distribution.
- [ ] A clean window and shared-host contamination check were recorded.
- [ ] Raw request evidence and environment evidence are durable and linked from
      the result.
- [ ] Tail latency, throughput, error rate, and resource cost were checked when
      they can trade off against the primary objective.
- [ ] `kernel_only_win` is not described as `end_to_end_win`.
