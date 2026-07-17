#!/usr/bin/env python3
"""Publish one verified main branch and annotated tag to two remotes."""

from __future__ import annotations

import os
import subprocess
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
