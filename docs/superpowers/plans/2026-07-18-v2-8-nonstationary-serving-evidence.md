# V2.8 nonstationary serving evidence implementation plan

## 1. Freeze the public contract

- Add strict design, series, and verdict schemas.
- Keep baseline/candidate roles, AB/BA block order, burn-in, minimum block count,
  metric taxonomy, and state tolerances closed.
- Document the trusted collector boundary and the no-performance-claim rule.

## 2. Write failing model tests

- Accept balanced AB/BA blocks whose timed state pairs meet every tolerance.
- Return `inconclusive_nonstationary` when a timed state pair crosses a frozen
  tolerance.
- Return inconclusive when burn-in-to-timed state shifts or row duration cross
  their separately frozen bounds.
- Reject a fixed-order or unbalanced design.
- Reject missing burn-in, reordered rows, undeclared state, and metric drift.
- Show that post-hoc row deletion cannot rescue a failed block.

## 3. Write failing CLI tests

- Rehash the raw series source without following symlinks.
- Reject a changed source, duplicate key, unsafe path, and plan mismatch.
- Emit deterministic JSON without running a benchmark or changing a host.

## 4. Implement the read-only guard

- Use the existing safe artifact helpers.
- Validate and canonicalize the design and series.
- Pair timed baseline/candidate rows by frozen block.
- Evaluate only predeclared state dimensions and tolerances.
- Emit a closed verdict and next-design recommendation.

## 5. Route the skill and documentation

- Run V2.8 after real workload collection and before a serving performance
  promotion decision.
- Add the reference, schemas, and CLI to `SKILL.md` and `self_check.py`.
- Update workflows and bilingual release notes without implying a universal
  performance result.

## 6. Verify and release

- Run focused tests, self-check, full CPU/static tests, and diff checks.
- Challenge the design with external AI and independent code review.
- Merge only with no Critical or Important findings.
- Publish `v2.8.0` to the fork and internal GitLab, then sync the installed skill.
