# CUDA Kernel Optimizer Legacy Coverage Native Port Design

**Date:** 2026-07-17

**Status:** Approved direction; written specification pending user review

**Target:** `skills/cuda-kernel-optimizer`

## Summary

Port the useful ideas from `origin/agent/complete-legacy-skill-coverage` onto
the current v2.2 architecture without merging its old state, ranking, or
promotion implementation.

The port has three independent surfaces:

1. concise serving and systems/IR evidence references;
2. a hardened standalone analyzer for an existing `.ncu-rep`; and
3. opt-in, workload-scoped strategy memory that remains advisory.

The authoritative v2.2 path remains unchanged: paired kernel evidence feeds
the inner decision, a user-owned workload feeds the outer decision, and only
`decision.json` may authorize promotion.

## Context and chosen approach

The legacy-coverage branch is one commit ahead of the pre-v2.2 base and 61
commits behind current `main`. A direct merge produces content conflicts in 11
core files, including `orchestrate.py`, `state.py`, `summarize.py`, `SKILL.md`,
and both READMEs. Its tests also exercise the removed median/noise/best-metric
promotion model.

Three approaches were considered:

1. **Merge the branch and resolve conflicts.** Rejected because it risks
   restoring pre-v2.2 promotion semantics and mixes unrelated features into
   security-sensitive lifecycle code.
2. **Copy only the two reference documents.** Low risk, but it leaves the
   useful report-analysis and cross-run-learning workflows unavailable.
3. **Reimplement each useful surface against v2.2 contracts.** Chosen because
   it preserves current evidence integrity and lets every new behavior be
   introduced behind focused tests.

## Goals

1. Let a user inspect an existing Nsight Compute report without creating or
   mutating an optimizer run.
2. Bind standalone report conclusions to stable report/source identities and
   preserve bounded raw evidence.
3. Let users explicitly retain evidence-backed optimization experience across
   identical frozen workloads.
4. Keep cross-run memory advisory and prevent it from blocking candidates or
   changing promotion.
5. Explain the evidence needed to move from generated code to operator,
   runtime, and serving claims.
6. Add systems, CUTLASS, and Triton IR coverage guidance through progressive
   disclosure instead of expanding the core `SKILL.md` substantially.

## Non-goals

- Merging or cherry-picking the legacy branch.
- Reintroducing average/median/noise-threshold promotion.
- Making strategy memory mandatory or enabling it by default.
- Automatically blacklisting a method because of a previous result.
- Treating a successfully imported `.ncu-rep` as proof of current counter
  permission, current source identity, or end-to-end performance.
- Replacing the v2.2 workload adapter, decision engine, state schema, artifact
  store, or checkpoint lifecycle.
- Adding new GPU-selection, tolerance, compiler, or profiler flags to the
  orchestrator in this change.

## Component 1: evidence references

Create two direct references from `SKILL.md`:

### `references/serving_evidence_protocol.md`

Define the claim ladder:

| Layer | Minimum evidence | Allowed claim |
|---|---|---|
| Generated code | bound compiler/SASS artifacts | the intended mechanism was emitted |
| Isolated operator | reference validation plus paired timing | the tested operator improved |
| Matched runtime | identical engine, inputs, cache policy, and runtime A/B | the implementation improved in that runtime |
| Serving endpoint | clean-window load test with identical model/request configuration | the tested endpoint metric improved |

The reference must route serving validation through the existing user-owned
workload contract. It must explicitly distinguish `kernel_only_win` from
`end_to_end_win`, explain shared-host contamination, and require raw request
and environment evidence before deployment claims.

### `references/systems_and_ir_coverage.md`

Document the evidence routing for:

- host/device copies, allocation, synchronization, CUDA Graphs, and launch
  density;
- CUTLASS/CuTe dispatch, layout, epilogue, cluster, and architecture routing;
- Triton autotune configurations, TTIR/TTGIR/LLVM/PTX, cache identity, and
  generated-code checks;
- sparse, variable-length, fused, and serving paths.

This reference links to `optimization_catalog.md`, `compatibility.md`, and the
serving protocol rather than duplicating their tables.

## Component 2: standalone `.ncu-rep` analysis

Create `scripts/analyze_ncu_rep.py` with this interface:

```text
python3 scripts/analyze_ncu_rep.py REPORT \
  --out-dir OUTPUT \
  [--source SOURCE] \
  [--ncu-bin NCU] \
  [--ncu-num 5] \
  [--timeout 120]
```

### Input rules

- `REPORT` must be an existing regular, non-symlink file whose parent path has
  no symlink components.
- `SOURCE`, when supplied, follows the same rule.
- `OUTPUT` must be a new or existing real directory with no symlink components.
- `ncu-num` and `timeout` must be positive finite values; booleans are invalid.
- Resolve the requested NCU executable once to a physical regular file, record
  both requested and resolved paths, and execute only that resolved identity.
  A normal toolchain symlink is allowed only when its final target is regular.

### Execution and identity

1. Open and hash the report with the secure artifact reader.
2. Optionally hash the source.
3. Import per-kernel summary, details, and CSV metrics with argv execution,
   bounded output, a hard timeout, and process-group cleanup.
4. Parse and rank metrics using the current `profile_ncu.py` classification
   behavior.
5. Re-open and re-hash the report and source; reject any identity drift.
6. Publish the output bundle atomically.

The analyzer never launches a target kernel and never changes driver or
counter policy. Importing a report therefore records `counter_access` as
`not_probed`.

### Output contract

Publish:

```text
OUTPUT/
├── analysis.json
├── analysis.md
├── raw.csv
├── summary.txt
├── summary.stderr.txt
├── details.txt
└── details.stderr.txt
```

`analysis.json` contains:

- schema version and terminal status (`success`, `partial`, or `failed`);
- report/source absolute paths and SHA-256 identities;
- resolved NCU identity and version when available;
- per-command return code, timeout flag, and truncation flag;
- kernel names, metric count, axis rankings, and heuristic primary axis;
- `counter_access: not_probed` and explicit interpretation limits; and
- hashes of every published evidence file.

`analysis.md` escapes all imported names and values so report content cannot
create Markdown links, images, headings, or tables. A partial import exits 2;
input/security/timeout failures exit nonzero and do not preserve a stale
`analysis.json` from an older run.

## Component 3: advisory strategy memory

Create `scripts/strategy_memory.py`. It is an explicit local tool, not a new
orchestrator stage.

### Interface

```text
python3 scripts/strategy_memory.py suggest \
  --memory MEMORY.json --manifest RUN/manifest.json --out SUGGESTION.json

python3 scripts/strategy_memory.py record \
  --memory MEMORY.json --run-dir RUN --out RECORD.json
```

No default memory path is used by the orchestrator. The user or agent must
explicitly select `--memory`; the core optimization run remains self-contained
when strategy memory is not requested.

### Scope identity

Derive the scope from the frozen v2.2 manifest, including:

- manifest schema version and input hash;
- backend and architecture;
- normalized dimensions and pointer size;
- baseline and reference SHA-256 identities; and
- workload source/objective identity, or an explicit kernel-only marker.

Hash the canonical scope document. Same filenames with different bytes must
never share a scope. Missing or malformed identity fields fail closed.

### Stored evidence

Use a versioned store with scopes, runs, method evidence, and bundle evidence.
`record` accepts only a completed v2.2 run. It must run the existing state and
decision schema validators, verify checkpoint/state/decision candidate
identity, and recompute every referenced raw-pair statistic before recording
the authoritative terminal decision and evidence hashes.

Individual methods receive positive/negative performance attribution only when
bound ablation evidence supports it. SASS evidence records implementation
status separately and cannot establish performance contribution. Without
ablation, the tool records only a bundle-level outcome. `inconclusive` remains
inconclusive.

### Advisory semantics

`suggest` returns:

- `preferred_method_ids`: repeated evidence-backed positive outcomes;
- `caution_method_ids`: negative, rejected, or conflicting outcomes;
- `prior_bundles`: exact bundle history; and
- evidence links and counts supporting every suggestion.

These are search hints. They must never remove a branch, override profiler
evidence, modify budget policy, or bypass current correctness, sanitizer,
paired, workload, or decision gates. `SKILL.md` instructs the agent to explain
when it retries a caution method because current bottleneck evidence changed.

### Storage safety

- Reject memory, lock, manifest, state, decision, and output paths containing
  symlink components.
- Use an adjacent lock file with `fcntl.flock` on supported Unix platforms.
- Read under the lock, update in memory, write a unique temporary regular file,
  `fsync` it, atomically replace the memory file, and `fsync` the directory.
- Create new memory files with mode `0600`.
- Allow at most 256 scopes and 128 run records per scope. Refuse a new unique
  record at the limit rather than silently evicting evidence. Deduplicate by
  run input hash, candidate hash, decision hash, and checkpoint completion
  identity.
- Refuse malformed, non-finite, unknown-version, or contradictory evidence.

## Skill and metadata changes

Keep `SKILL.md` under its current progressive-disclosure structure:

- mention standalone report analysis in the profiler entry point;
- after finalization, mention optional strategy-memory record/suggest commands;
- link the two new references only when the task involves runtime/serving or
  systems/IR analysis; and
- restate that memory suggestions are advisory and promotion remains owned by
  `decision.json`.

Update `agents/openai.yaml` only if its description/default prompt no longer
mentions the expanded `.ncu-rep` and serving triggers. Update both READMEs with
short executable examples and links; detailed rules remain in references and
`--help` output.

## Error handling

- Existing reports may be imported with partial metric coverage, but missing
  or unsafe inputs are hard failures.
- Timeouts kill the entire import process group and produce bounded diagnostic
  output without a success artifact.
- Strategy memory corruption, version drift, symlinks, concurrency failures,
  or evidence contradictions fail closed and leave the previous valid store
  intact.
- An unavailable memory store never degrades or blocks the optimizer run;
  report it as unavailable and continue without memory suggestions.
- Neither tool logs environment secrets, arbitrary environment variables, or
  unbounded command output.

## Test strategy

Use strict red-green-refactor cycles.

### Report analyzer tests

- regular report success with a fake NCU importer;
- partial import and exact exit status;
- timeout with a term-ignoring child process;
- report/source same-size same-mtime drift detection;
- report, source, output, and executable symlink rejection;
- stale output removal and atomic bundle behavior;
- bounded stdout/stderr and hostile Markdown escaping;
- finite numeric validation and report identity binding.

### Strategy memory tests

- identical frozen inputs produce the same scope;
- different source bytes with identical filenames/dimensions produce different
  scopes;
- completed, fully bound decision evidence records once;
- incomplete, contradictory, tampered, legacy, and non-finite evidence fail;
- individual method attribution requires bound evidence;
- concurrent writers preserve both records;
- symlink and path-replacement attacks fail without overwriting the target;
- suggestions are advisory and never mutate manifests, state, or decisions.

### Documentation and regression tests

- future agents can find both new commands and references from `SKILL.md`;
- README examples match the CLI help;
- old pending/V2.1 claims do not return;
- the full existing CPU suite remains green;
- `quick_validate.py`, all script `--help` commands, `py_compile`, and
  `git diff --check` pass.

Forward validation should use a copied `.ncu-rep` under the isolated RTX 5090
test root. It must not change counter permissions, driver configuration, or
the user's source trees.

## Rollout and repository boundaries

- Develop on `agent/legacy-coverage-v2-2` in an isolated worktree.
- Commit the approved specification before writing the implementation plan.
- Implement each component independently with focused commits and full
  regression at the end.
- Do not merge the legacy branch and do not push to the upstream repository.
- Keep changes local until the user explicitly approves fork integration and
  push for this follow-up.
