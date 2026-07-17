# V2.5 Migration Notes

V2.4.1 manifests and controller inputs remain accepted by their existing APIs.
V2.5 adds a separate formal evidence layer; it does not silently reinterpret old
runs. There is no in-place evidence rewrite.

## Compatibility result

A read-only import of an old run returns `legacy_unsealed` with
`evidence_integrity=UNKNOWN`. It remains valid historical V2.4.1 evidence at its
original claim layer, but it cannot become a V2.5 promotion result merely by
adding hashes after the fact.

The following remain unchanged:

- kernel-loop V2 manifest and state/checkpoint formats;
- `cuda-workload-optimizer/control-v1` and existing ChangeSet/reviewer formats;
- the existing workload objective and adapter contracts;
- V2.4.1 default workload evaluation when no V2.5 design is supplied.

## Create a new formal attempt

Do not mutate the old directory. Create a new attempt and supply:

1. a frozen experiment design with a complete balanced schedule, statistical
   units, CI, pair/win minimums, relative and absolute guardrails,
   no-exclusion, and whole-pair-only pre-measurement retry policy;
2. a shared-host guard policy plus continuous samples and phase markers;
3. an execution-path proof for every expected case;
4. separate source, binary, plugin, engine, backend, server, image, tactic, and
   timing-cache artifact identities required by the claim layer;
5. a frozen serving experiment for HTTP/gRPC c1/c2/c4/c8/c12 when claiming an
   endpoint result;
6. raw formal rows and a separate performance verdict;
7. a terminal attempt state followed by seal, audit, decision, and the final
   evidence manifest.

Formal `workload_evaluate.evaluate_pairs(..., experiment_design=...)` requires
`retries=0` and uses the exact frozen schedule. Calls without
`experiment_design` keep V2.4.1 behavior and are not V2.5-sealed formal runs.

## Imported evidence

Use `scripts/evidence.py audit-imported` with an output directory outside the
imported tree. The command never launches the serving workload, Nsys, or NCU and
never changes the imported evidence. Missing guard, identity, coverage, or raw
artifacts remain missing.
