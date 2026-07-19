# Triton decode attention / GQA skill TDD

Date: 2026-07-19

Scenario: RTX 5090 / `sm_120`, Q heads 32, KV heads 8, head dimension 128,
batch 1–16, context 128–8192. Short contexts show launch gaps; long contexts
show KV gather and DRAM pressure; non-128-aligned tails fail the PyTorch SDPA
oracle. NCU counters are unavailable. Budget: 12 candidates and 45 GPU minutes.

## Baseline without the playbook

The baseline correctly refused promotion and put correctness first. It also:

- loaded the main router plus workload controller, direction admission,
  performance iteration, offline knowledge, compatibility and limits;
- proposed grouping four Q heads under one KV-head program before proving that
  the current implementation actually repeats K/V work;
- acknowledged that register pressure and spill behavior were unknown;
- invented a 6 minute 45 second infrastructure allowance that was not present
  in the contract.

This established the behavior the playbook needed to change.

## First forward run

The model queried metadata, loaded only `triton.decode-attention-gqa`, and chose
a bounded tail-mask correctness repair as C0. It refused timing and promotion,
kept launch and KV-reuse mechanisms separate, treated DRAM reduction as an
unverified hypothesis without NCU, and used only the stated 12-candidate / 45
GPU-minute budget.

The first query still exposed `candidate_admissible=true`. External challenge
identified that as authority leakage and a semantic deadlock: retrieval must
not decide whether a candidate may execute.

## Refactor gate

The refactored query:

- matches complete signal groups instead of any single broad signal;
- returns `retrieval_status`, never `candidate_admissible`;
- reports `execution_authority=none`;
- carries distinct `pre_execution` and `promotion` gates for the Controller;
- rejects a registry whose declared context cost differs from the exact UTF-8
  byte length of the hash-bound playbook.

This file records model behavior, not performance proof. GPU and workload
benefit still require the five-arm evaluation and RTX 5090 run.
