# Standalone V1 release implementation plan

> **For AI agents:** Required sub-skill: use superpowers:test-driven-development
> for behavior changes and superpowers:verification-before-completion before
> every commit, push, release, or archive action. Track progress with checkboxes.

**Goal:** Publish a clean, independently maintained `cuda-kernel-optimizer`
GitHub repository and retire the old public fork without losing history.

**Architecture:** The current repository and unchanged internal GitLab remote
retain full history. A filtered filesystem snapshot becomes a new Git root.
The standalone repository owns future development, CI, releases, and community
workflows.

**Technical stack:** Git, GitHub CLI, GitHub Actions, Python unittest, Codex
skill package, deterministic ZIP release artifact.

---

### Task 1: Close the historical implementation

**Files:**
- Modify: the eight currently changed Controller and regression-test files
- Create: `maintainers/standalone-release-design.md`
- Create: `maintainers/standalone-release-plan.md`

- [ ] Run `python3 -m unittest discover -s tests -p 'test_*.py'` and require zero failures.
- [ ] Run `python3 -m unittest discover -s skills/cuda-kernel-optimizer/tests -p 'test_*.py'`.
- [ ] Run `python3 -m compileall -q skills/cuda-kernel-optimizer/scripts tests` and `git diff --check`.
- [ ] Commit the Controller fixes separately from the standalone-release design.
- [ ] Push the completed historical branch to GitHub and GitLab without force.
- [ ] Merge it into both historical `main` branches only after commit identities are verified.

### Task 2: Create the standalone snapshot

**Files:**
- Create repository directory: `../cuda-kernel-optimizer`
- Remove from snapshot: `maintainers/`, `tools/publish_dual_remote.py`, `tests/test_publish_dual_remote.py`
- Remove unused assets only after `rg` proves there are no references

- [ ] Export tracked files from the verified historical commit into a new directory.
- [ ] Remove the excluded paths and all old Git metadata.
- [ ] Initialize a new Git repository with `main` as the default branch.
- [ ] Verify the new repository has no parent commit and no remote.

### Task 3: Define standalone project contracts with failing tests

**Files:**
- Create: `tests/test_standalone_project.py`
- Create: `VERSION`
- Create: `NOTICE`
- Create: `CONTRIBUTING.md`
- Create: `SECURITY.md`
- Create: `.github/workflows/ci.yml`
- Create: `.github/ISSUE_TEMPLATE/bug_report.yml`
- Create: `.github/ISSUE_TEMPLATE/feature_request.yml`
- Create: `.github/pull_request_template.md`
- Modify: `README.md`, `README.zh-CN.md`, `docs/validation.md`

- [ ] Add tests requiring version `1.0.0`, the new repository URL, origin notice,
      CI matrix, contribution/security files, no maintainer directory, no
      obsolete dual publisher, and no old public installation URL.
- [ ] Run `python3 -m unittest tests.test_standalone_project` and confirm it fails
      because the standalone files and metadata are absent.
- [ ] Add the minimum files and README changes that satisfy the contract.
- [ ] Run the standalone contract test and existing public-document tests until green.

### Task 4: Validate the complete standalone tree

**Files:**
- Validate all tracked files in the standalone repository

- [ ] Run the full root test suite and the skill-local test suite.
- [ ] Run compileall, `git diff --check`, link/README consistency tests, and `self_check.py`.
- [ ] Build `cuda-kernel-optimizer-v1.0.0.zip` from only
      `skills/cuda-kernel-optimizer/` and generate its SHA-256 file.
- [ ] Extract the ZIP into a temporary Codex skills root and run `self_check.py`
      from the extracted package.
- [ ] Commit the verified tree as the single root commit.

### Task 5: Publish and protect the new GitHub repository

**Files:**
- GitHub repository: `troycheng/cuda-kernel-optimizer`

- [ ] Create a public repository with Issues enabled and the approved About and topics.
- [ ] Push `main`, confirm GitHub reports `fork=false`, and wait for both CI matrix jobs.
- [ ] Fresh-clone the GitHub repository and rerun the standalone contract and self-check.
- [ ] Protect `main` from force pushes and deletion and require the CI check for future changes.
- [ ] Create tag and release `v1.0.0`; attach the ZIP and SHA-256 file.
- [ ] Download the release assets and verify their checksum and extracted self-check.

### Task 6: Retire the old public fork safely

**Files:**
- Modify in old GitHub fork only: `README.md`, `README.zh-CN.md`

- [ ] Add a prominent migration notice linking to the verified standalone repository.
- [ ] Push the notice without changing the internal GitLab history repository.
- [ ] Confirm the old and new URLs, release asset, and installation text resolve.
- [ ] Archive `troycheng/cuda-optimized-skill` through the GitHub API.
- [ ] Confirm it is archived and the new repository remains public, writable, and non-fork.

### Task 7: Install and hand off

**Files:**
- Update: `~/.codex/skills/cuda-kernel-optimizer/`

- [ ] Synchronize the installed skill from the standalone repository.
- [ ] Verify a dry-run sync reports no differences and production-script checksums match.
- [ ] Record final repository, commit, CI, release, archive, and local-install evidence.
