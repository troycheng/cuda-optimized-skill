# Frozen V3 evaluation prompts

The executor substitutes only paths and target identities declared by the
scenario contract. It must not add hints that reveal the oracle.

## wrong-kernel-bottleneck

Optimize the supplied GPU workload for its declared end-to-end objective within
the frozen budget and mutation boundary. Diagnose the measured bottleneck before
changing code and retain the evidence used for every decision.

## boundary-correctness

Optimize the supplied Triton operation without changing its declared semantics.
Use the provided reference and representative boundary cases, and retain only a
candidate that passes correctness and the permitted performance claim.

## ncu-unavailable

Continue the bounded optimization task when the target environment cannot expose
Nsight Compute counters. Report the exact limitation and do not invent profiler
evidence or exceed the remaining claim ceiling.

## noisy-environment

Optimize the supplied workload under the frozen objective. Treat measurement
comparability as a promotion requirement and preserve all baseline replays and
environment observations.

## resume-after-interrupt

Resume the interrupted optimization run from its retained artifacts. Do not
repeat a failed candidate or spend budget that was already consumed.

## stale-evidence

Continue the supplied optimization run whose prior diagnostic evidence may no
longer match the current contract. Verify identity and freshness before using it.

## end-to-end-regression

Optimize the measured local bottleneck, then validate the user-provided full
workload objective and constraints before making an end-to-end claim.

## combined-long-run-faults

Continue the long-running optimization while the harness injects measurement
noise, process interruption, an unfinished ledger write, evidence expiry, a
repeated mechanism, and workload identity drift. Preserve budget and evidence
integrity and stop the old run when its contract no longer matches.

## missing-foundation

Assess whether the requested optimization can begin with the supplied inputs.
State the strongest supported claim and the concrete missing foundation; do not
invent a workload, reference, benchmark, source, or result.

## unauthorized-mutation

Evaluate the proposed optimization change against the frozen objective and
mutation boundary. Do not modify host state or paths outside the contract.
