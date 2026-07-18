# Iteration {{iter}} — V2.6 performance decision record

## Performance hypothesis

- **Statement**: {{hypothesis_statement}}
- **Mechanism**: {{hypothesis_mechanism}}
- **Target / direction / minimum effect**: {{target_metric}} /
  {{target_direction}} / {{minimum_effect_pct}}%
- **Authorized mutation scope**: {{mutation_scope}}
- **Prevalidated measurement path**: {{measurement_path_id}}@
  {{measurement_path_version}}

## Candidate and result

- **Candidate / changed paths**: {{candidate_id}} / {{changed_paths}}
- **Correctness**: {{correctness_result}}
- **Performance result**: {{performance_result}}
- **Work class**: {{candidate_evaluated_or_other}}

`candidate_evaluated` means a real candidate completed its applicable gates. A
loss, inconclusive result, or correctness failure is not a performance gain.

## Decision and next performance action

- **Keep or reject**: {{keep_or_reject}}
- **Reason**: {{decision_reason}}
- **Next performance action**: {{next_performance_action}}

## Measurement blocker, if any

- **Blocker**: {{measurement_blocker_or_none}}
- **Infrastructure time / cap**: {{infrastructure_seconds}} /
  {{infrastructure_cap_seconds}} seconds
- **Repairs / cap**: {{infrastructure_repairs}} / 1
- **Fallback or stop**: {{fallback_or_stop}}

Infrastructure work is not an optimization result. Report this section only
when it blocks candidate evaluation.

## Frozen context

- **Run input hash**: `{{input_hash}}`
- **Mode**: {{mode}} (`kernel-only` / `full`)
- **Budget preset**: {{budget_name}}
- **Current best**: `{{best_file_before}}`
- **GPU / architecture**: {{gpu_name}} / {{sm_arch}}
- **Reference**: `{{ref_file}}`
- **User-provided workload**: {{workload_source_or_none}}

## Available evidence

- **NCU counter access**: {{ncu_counter_status}}
- **NCU failure, if any**: {{ncu_error_or_none}}
- **Sanitizer policy / coverage**: {{sanitizer_mode}} / {{sanitizer_coverage}}
- **Compiler evidence**: {{compiler_evidence_status}}
- **SASS evidence**: {{sass_evidence_status}}

Missing or degraded evidence stays missing or degraded. In particular,
`ERR_NVGPUCTRPERM` is not a reason to add privileges automatically.
Compiler and SASS findings are evidence and method-classification inputs, not
hard promotion gates. Record source or artifact identity drift as an integrity
failure rather than treating it as ordinary missing coverage.

## Bottleneck evidence

| Axis | Observed evidence | Gap | Method budget |
|---|---|---:|---:|
| Compute | {{compute_evidence}} | {{delta_compute}} | {{budget_compute}} |
| Memory | {{memory_evidence}} | {{delta_memory}} | {{budget_memory}} |
| Latency | {{latency_evidence}} | {{delta_latency}} | {{budget_latency}} |

- **Analysis model / quality**: {{analysis_model}} / {{analysis_quality}}
- **Primary bound, if evidenced**: {{bound}}
- **Missing evidence**: {{missing_evidence}}

## Method selection

For every selected method, cite an observable trigger, compatibility rule,
implementation delta, expected effect, and rejection criterion. List each
higher-priority method that was scanned and skipped. Do not include hidden
reasoning.

| Method id | Priority | Trigger evidence | Implementation delta | Reject when |
|---|---:|---|---|---|
| {{method_1_id}} | {{method_1_priority}} | {{method_1_trigger}} | {{method_1_delta}} | {{method_1_reject}} |
| {{method_2_id}} | {{method_2_priority}} | {{method_2_trigger}} | {{method_2_delta}} | {{method_2_reject}} |

### Excluded candidates

| Method id | Priority | Exact skip reason |
|---|---:|---|
| {{excluded_method_id}} | {{excluded_priority}} | {{excluded_reason}} |

### Orthogonality and compatibility

- Architecture/public API compatibility: {{compatibility_check}}
- Coupled or duplicate methods excluded: {{orthogonality_check}}
- Previously ineffective/failed methods handled: {{history_check}}

## Branch plan

All branches implement the selected method intent; vary only bounded
hyperparameters or implementation details.

| Branch | Tile/config | Stages | Warps | Other delta |
|---|---|---:|---:|---|
| b1 | {{b1_tile}} | {{b1_stages}} | {{b1_warps}} | {{b1_delta}} |
| b2 | {{b2_tile}} | {{b2_stages}} | {{b2_warps}} | {{b2_delta}} |

## Inner kernel evidence

Fill this section from durable artifacts after `close-iter`.
Record the actual lifecycle: correctness → paired → sanitizer → SASS. The
sanitizer processes the statistically confirmed shortlist; SASS describes the
final eligible candidate.

| Candidate | Correctness | Sanitizer | Estimate | Confidence interval | Valid/invalid pairs | Verdict |
|---|---|---|---:|---|---:|---|
| {{candidate_id}} | {{correctness}} | {{sanitizer_status}} | {{kernel_estimate_pct}}% | [{{kernel_ci_low_pct}}%, {{kernel_ci_high_pct}}%] | {{valid_pairs}} / {{invalid_pairs}} | {{kernel_verdict}} |

- **Kernel raw pairs**: `{{kernel_paired_samples_path}}`
- **Compiler manifest**: `{{compiler_manifest_path}}`
- **SASS result**: `{{sass_result_path}}`

Only `confirmed_win` can enter the outer shortlist. `inconclusive` does not
promote.

## Outer workload evidence

- **Status**: {{workload_status_or_not_supplied}}
- **Primary metric**: {{primary_metric}} ({{primary_direction}})
- **Estimate / confidence interval**: {{workload_estimate_pct}}% /
  [{{workload_ci_low_pct}}%, {{workload_ci_high_pct}}%]
- **Minimum effect**: {{workload_min_effect_pct}}%
- **Constraint results**: {{constraint_results}}
- **Workload raw pairs**: `{{workload_paired_samples_path_or_none}}`

Without a user-provided workload, write “not supplied; end-to-end claim is not
available.”

## Terminal decision

- **Decision artifact**: `{{decision_path}}`
- **Outcome**: {{terminal_outcome}}
- **Promotion**: {{promoted_or_retained}}
- **Reason**: {{decision_reason}}
- **Next checkpoint stage**: {{next_stage}}

`kernel_only_win` confirms only the kernel result. It is normal in kernel-only
mode and can also be terminal in full mode after workload
failure/loss/inconclusive evidence; in full mode it must retain the global best.
`end_to_end_win` requires confirmed kernel and workload wins plus passing
constraints and is the only full-mode result that may promote the global best.
