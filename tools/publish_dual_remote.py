#!/usr/bin/env python3
"""Publish one verified main branch and annotated tag to two remotes."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ReleaseTargets:
    github_remote: str
    github_url: str
    internal_remote: str
    internal_url: str
    upstream_remote: str
    upstream_fetch_url: str
    upstream_push_url: str = "DISABLED"


PRODUCTION_TARGETS = ReleaseTargets(
    github_remote="origin",
    github_url="https://github.com/troycheng/cuda-optimized-skill.git",
    internal_remote="internal",
    internal_url="git@git.yukework.com:mlsys/cuda-optimized-skill.git",
    upstream_remote="upstream",
    upstream_fetch_url="https://github.com/KernelFlow-ops/cuda-optimized-skill.git",
)


@dataclass(frozen=True)
class ReleaseIdentity:
    main_commit: str
    tag_name: str
    tag_object: str
    tag_commit: str


@dataclass(frozen=True)
class RemoteState:
    main_commit: str | None
    tag_object: str | None
    tag_commit: str | None


class PublishError(RuntimeError):
    def __init__(self, message: str, *, status: str = "not_started") -> None:
        super().__init__(message)
        self.status = status


def _git(
    repo: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and completed.returncode != 0:
        command = "git " + " ".join(args)
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PublishError(f"{command} failed: {detail}")
    return completed


def _remote_urls(repo: Path, remote: str, *, push: bool) -> list[str]:
    args: list[str] = ["remote", "get-url"]
    if push:
        args.append("--push")
    args.extend(["--all", remote])
    result = _git(repo, *args)
    return [line for line in result.stdout.splitlines() if line]


def _require_remote_urls(
    repo: Path,
    remote: str,
    *,
    fetch_url: str,
    push_url: str,
) -> None:
    fetch_urls = _remote_urls(repo, remote, push=False)
    push_urls = _remote_urls(repo, remote, push=True)
    if fetch_urls != [fetch_url]:
        raise PublishError(f"unexpected fetch URL for {remote}")
    if push_urls != [push_url]:
        raise PublishError(f"unexpected push URL for {remote}")


def _validate_remote_identity(repo: Path, targets: ReleaseTargets) -> None:
    _require_remote_urls(
        repo,
        targets.github_remote,
        fetch_url=targets.github_url,
        push_url=targets.github_url,
    )
    _require_remote_urls(
        repo,
        targets.internal_remote,
        fetch_url=targets.internal_url,
        push_url=targets.internal_url,
    )
    _require_remote_urls(
        repo,
        targets.upstream_remote,
        fetch_url=targets.upstream_fetch_url,
        push_url=targets.upstream_push_url,
    )


def _clean_output(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout.strip()


def inspect_local(repo: Path, tag: str, targets: ReleaseTargets) -> ReleaseIdentity:
    repo = repo.resolve()
    branch = _clean_output(_git(repo, "branch", "--show-current"))
    if branch != "main":
        raise PublishError("release must run from the main branch")
    if _clean_output(_git(repo, "status", "--porcelain", "--untracked-files=all")):
        raise PublishError("release requires a clean worktree and index")

    _validate_remote_identity(repo, targets)

    ref = f"refs/tags/{tag}"
    valid_ref = _git(repo, "check-ref-format", ref, check=False)
    if valid_ref.returncode != 0:
        raise PublishError("invalid release tag name")
    object_type = _git(repo, "cat-file", "-t", ref, check=False)
    if object_type.returncode != 0:
        raise PublishError(f"release tag does not exist: {tag}")
    if _clean_output(object_type) != "tag":
        raise PublishError("release tag must be annotated")

    main_commit = _clean_output(_git(repo, "rev-parse", "refs/heads/main"))
    tag_object = _clean_output(_git(repo, "rev-parse", ref))
    tag_commit = _clean_output(_git(repo, "rev-parse", f"{ref}^{{commit}}"))
    reachable = _git(
        repo,
        "merge-base",
        "--is-ancestor",
        tag_commit,
        main_commit,
        check=False,
    )
    if reachable.returncode != 0:
        raise PublishError("release tag commit must be reachable from main")
    return ReleaseIdentity(
        main_commit=main_commit,
        tag_name=tag,
        tag_object=tag_object,
        tag_commit=tag_commit,
    )


def _parse_ls_remote(lines: Sequence[str], tag: str) -> RemoteState:
    refs: dict[str, str] = {}
    for line in lines:
        fields = line.split("\t", 1)
        if len(fields) != 2:
            raise PublishError("malformed git ls-remote output")
        commit, ref = fields
        refs[ref] = commit
    return RemoteState(
        main_commit=refs.get("refs/heads/main"),
        tag_object=refs.get(f"refs/tags/{tag}"),
        tag_commit=refs.get(f"refs/tags/{tag}^{{}}"),
    )


def inspect_remote(repo: Path, remote: str, tag: str) -> RemoteState:
    result = _git(
        repo,
        "ls-remote",
        remote,
        "refs/heads/main",
        f"refs/tags/{tag}",
        f"refs/tags/{tag}^{{}}",
    )
    return _parse_ls_remote(result.stdout.splitlines(), tag)


def ensure_fast_forward(
    repo: Path,
    remote: str,
    local: ReleaseIdentity,
    remote_state: RemoteState,
) -> None:
    if remote_state.tag_object is not None:
        if (
            remote_state.tag_object != local.tag_object
            or remote_state.tag_commit != local.tag_commit
        ):
            raise PublishError(f"conflicting release tag on {remote}")

    if remote_state.main_commit in (None, local.main_commit):
        return

    fetched = _git(
        repo,
        "fetch",
        "--no-tags",
        remote,
        "refs/heads/main",
        check=False,
    )
    if fetched.returncode != 0:
        raise PublishError(f"cannot fetch current main from {remote}")
    is_ancestor = _git(
        repo,
        "merge-base",
        "--is-ancestor",
        remote_state.main_commit,
        local.main_commit,
        check=False,
    )
    if is_ancestor.returncode != 0:
        raise PublishError(f"main on {remote} cannot be fast-forwarded")


def _run_validation(repo: Path) -> None:
    commands = (
        (
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
            "-q",
        ),
        (sys.executable, "-m", "unittest", "tests.test_skill_metadata", "-q"),
    )
    for command in commands:
        completed = subprocess.run(
            list(command),
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode != 0:
            output = "\n".join(completed.stdout.splitlines()[-40:])
            raise PublishError(f"release validation failed:\n{output}")


def _verify_exact(remote: str, identity: ReleaseIdentity, state: RemoteState) -> None:
    if state.main_commit != identity.main_commit:
        raise PublishError(f"main verification failed on {remote}")
    if state.tag_object != identity.tag_object:
        raise PublishError(f"tag object verification failed on {remote}")
    if state.tag_commit != identity.tag_commit:
        raise PublishError(f"tag commit verification failed on {remote}")


def _push_release(
    repo: Path,
    remote: str,
    identity: ReleaseIdentity,
    *,
    failure_status: str,
) -> None:
    pushed = _git(
        repo,
        "push",
        "--atomic",
        remote,
        "refs/heads/main:refs/heads/main",
        f"refs/tags/{identity.tag_name}:refs/tags/{identity.tag_name}",
        check=False,
    )
    if pushed.returncode != 0:
        detail = pushed.stderr.strip() or pushed.stdout.strip()
        raise PublishError(
            f"push to {remote} failed: {detail}",
            status=failure_status,
        )


def _summary(
    status: str,
    identity: ReleaseIdentity,
    *,
    github: str,
    internal: str,
) -> dict[str, object]:
    return {
        "status": status,
        "main_commit": identity.main_commit,
        "tag": {
            "name": identity.tag_name,
            "object": identity.tag_object,
            "commit": identity.tag_commit,
        },
        "remotes": {
            "github": github,
            "internal": internal,
        },
    }


def publish(
    repo: Path,
    tag: str,
    targets: ReleaseTargets,
    *,
    execute: bool,
    validate: bool = True,
) -> dict[str, object]:
    repo = repo.resolve()
    identity = inspect_local(repo, tag, targets)
    github_before = inspect_remote(repo, targets.github_remote, tag)
    ensure_fast_forward(repo, targets.github_remote, identity, github_before)
    internal_before = inspect_remote(repo, targets.internal_remote, tag)
    ensure_fast_forward(repo, targets.internal_remote, identity, internal_before)

    if validate:
        _run_validation(repo)
    if not execute:
        return _summary(
            "dry_run",
            identity,
            github="planned",
            internal="planned",
        )

    _push_release(
        repo,
        targets.github_remote,
        identity,
        failure_status="github_failed",
    )
    try:
        github_after = inspect_remote(repo, targets.github_remote, tag)
        _verify_exact(targets.github_remote, identity, github_after)
    except PublishError as exc:
        raise PublishError(str(exc), status="github_failed") from exc

    _push_release(
        repo,
        targets.internal_remote,
        identity,
        failure_status="internal_pending",
    )
    try:
        internal_after = inspect_remote(repo, targets.internal_remote, tag)
        _verify_exact(targets.internal_remote, identity, internal_after)
    except PublishError as exc:
        raise PublishError(str(exc), status="internal_pending") from exc

    return _summary(
        "complete",
        identity,
        github="verified",
        internal="verified",
    )


def _redact(message: str) -> str:
    return re.sub(r"(?i)(https?://)([^/@\s]+)@", r"\1***@", message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish main and one annotated release tag to the GitHub source "
            "and internal GitLab mirror. The default is dry-run."
        )
    )
    parser.add_argument("--tag", required=True, help="annotated release tag")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform both remote pushes after all checks pass",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    repo: Path | None = None,
    targets: ReleaseTargets = PRODUCTION_TARGETS,
    validate: bool = True,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = publish(
            repo or Path.cwd(),
            args.tag,
            targets,
            execute=args.execute,
            validate=validate,
        )
    except PublishError as exc:
        result = {"status": exc.status, "error": _redact(str(exc))}
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
