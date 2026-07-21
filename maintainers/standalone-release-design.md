# Standalone V1 release design

## Goal

Publish `troycheng/cuda-kernel-optimizer` as a public, non-fork GitHub project
with a clean Git root and a `v1.0.0` release. The existing GitLab repository
remains the complete development-history source. The old GitHub fork is
archived only after the new project is independently verified.

## Repository identity

- New maintained project: `github.com/troycheng/cuda-kernel-optimizer`.
- Historical source: `git.yukework.com/mlsys/cuda-optimized-skill`.
- Old public fork: `github.com/troycheng/cuda-optimized-skill`; add a migration
  notice, then archive it.
- The new project starts from one root commit. It does not import old commits,
  branches, tags, releases, or maintainer work logs.
- The public project version starts at `v1.0.0`. Existing schema and protocol
  versions remain unchanged because they are compatibility identifiers.

## Published tree

Keep the runtime skill, tests, user documentation, brand assets, license, and
small release tools. Add CI, contribution guidance, a security policy, issue
templates, a version file, and an origin notice.

Exclude maintainer plans and research logs, the obsolete dual-remote publisher,
old release history, unused historical images, and links to the old GitHub fork.
Do not refactor runtime controllers or remove schemas during this release pass.

The public user path stays small:

1. `README.md` and `README.zh-CN.md` explain purpose, limits, and installation.
2. `skills/cuda-kernel-optimizer/SKILL.md` is the AI execution protocol.
3. `skills/cuda-kernel-optimizer/scripts/self_check.py` verifies an installation
   without claiming that a GPU workload has been tested.

## Verification and release

CI runs the CPU/static suite on Python 3.10 and 3.12. Physical GPU tests remain
explicitly opt-in. Before release, a fresh clone must pass the complete local
suite, skill self-check, documentation checks, release-tree checks, and release
archive checksum verification.

The release sequence is fail-safe: finish the historical commit, create and
test the standalone snapshot, publish the new repository, wait for CI, verify a
fresh clone and the `v1.0.0` asset, then update and archive the old GitHub fork.
No repository is deleted or force-pushed. If any step fails, the old GitHub
repository remains active.
