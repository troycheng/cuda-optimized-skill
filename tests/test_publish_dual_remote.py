import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.publish_dual_remote import (
    PublishError,
    ReleaseTargets,
    ensure_fast_forward,
    inspect_local,
    inspect_remote,
    main,
    publish,
)


def git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


class RepositoryCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.repo = self.root / "work"
        self.github = self.root / "github.git"
        self.internal = self.root / "internal.git"
        self.upstream = self.root / "upstream.git"

        git(self.root, "init", "--bare", str(self.github))
        git(self.root, "init", "--bare", str(self.internal))
        git(self.root, "init", "--bare", str(self.upstream))
        git(self.root, "init", "-b", "main", str(self.repo))
        git(self.repo, "config", "user.name", "Test User")
        git(self.repo, "config", "user.email", "test@example.com")
        (self.repo / "README.md").write_text("baseline\n", encoding="utf-8")
        git(self.repo, "add", "README.md")
        git(self.repo, "commit", "-m", "baseline")
        git(self.repo, "tag", "-a", "v1.0.0", "-m", "v1.0.0")

        git(self.repo, "remote", "add", "origin", str(self.github))
        git(self.repo, "remote", "add", "internal", str(self.internal))
        git(self.repo, "remote", "add", "upstream", str(self.upstream))
        git(self.repo, "remote", "set-url", "--push", "upstream", "DISABLED")
        self.targets = ReleaseTargets(
            github_remote="origin",
            github_url=str(self.github),
            internal_remote="internal",
            internal_url=str(self.internal),
            upstream_remote="upstream",
            upstream_fetch_url=str(self.upstream),
        )

    def push_release(self, remote: str = "origin") -> None:
        git(
            self.repo,
            "push",
            "--atomic",
            remote,
            "refs/heads/main:refs/heads/main",
            "refs/tags/v1.0.0:refs/tags/v1.0.0",
        )

    def clone_remote(self, remote: Path, name: str) -> Path:
        clone = self.root / name
        git(self.root, "clone", str(remote), str(clone))
        git(clone, "config", "user.name", "Remote User")
        git(clone, "config", "user.email", "remote@example.com")
        return clone

    def reject_pushes(self, remote: Path) -> None:
        hook = remote / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        hook.chmod(0o755)


class PreflightTests(RepositoryCase):
    def test_accepts_empty_remote_and_annotated_reachable_tag(self) -> None:
        identity = inspect_local(self.repo, "v1.0.0", self.targets)
        remote = inspect_remote(self.repo, "origin", "v1.0.0")

        self.assertEqual(git(self.repo, "rev-parse", "main").stdout.strip(), identity.main_commit)
        self.assertEqual("tag", git(self.repo, "cat-file", "-t", identity.tag_object).stdout.strip())
        self.assertEqual(identity.main_commit, identity.tag_commit)
        self.assertIsNone(remote.main_commit)
        ensure_fast_forward(self.repo, "origin", identity, remote)

    def test_rejects_dirty_tree(self) -> None:
        (self.repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")

        with self.assertRaises(PublishError) as caught:
            inspect_local(self.repo, "v1.0.0", self.targets)

        self.assertEqual("not_started", caught.exception.status)
        self.assertIn("clean", str(caught.exception).lower())

    def test_rejects_lightweight_tag(self) -> None:
        git(self.repo, "tag", "v1-light")

        with self.assertRaises(PublishError) as caught:
            inspect_local(self.repo, "v1-light", self.targets)

        self.assertEqual("not_started", caught.exception.status)
        self.assertIn("annotated", str(caught.exception).lower())

    def test_matching_remote_is_idempotent(self) -> None:
        self.push_release()
        identity = inspect_local(self.repo, "v1.0.0", self.targets)
        remote = inspect_remote(self.repo, "origin", "v1.0.0")

        ensure_fast_forward(self.repo, "origin", identity, remote)

        self.assertEqual(identity.main_commit, remote.main_commit)
        self.assertEqual(identity.tag_object, remote.tag_object)
        self.assertEqual(identity.tag_commit, remote.tag_commit)

    def test_rejects_remote_main_that_is_ahead(self) -> None:
        self.push_release()
        clone = self.clone_remote(self.github, "ahead")
        (clone / "remote.txt").write_text("remote\n", encoding="utf-8")
        git(clone, "add", "remote.txt")
        git(clone, "commit", "-m", "remote commit")
        git(clone, "push", "origin", "main")
        identity = inspect_local(self.repo, "v1.0.0", self.targets)
        remote = inspect_remote(self.repo, "origin", "v1.0.0")

        with self.assertRaises(PublishError) as caught:
            ensure_fast_forward(self.repo, "origin", identity, remote)

        self.assertEqual("not_started", caught.exception.status)
        self.assertIn("fast-forward", str(caught.exception).lower())

    def test_rejects_conflicting_remote_tag(self) -> None:
        self.push_release()
        clone = self.clone_remote(self.github, "tag-conflict")
        (clone / "remote.txt").write_text("remote\n", encoding="utf-8")
        git(clone, "add", "remote.txt")
        git(clone, "commit", "-m", "remote commit")
        git(clone, "tag", "-f", "-a", "v1.0.0", "-m", "conflict")
        git(clone, "push", "--force", "origin", "refs/tags/v1.0.0")
        identity = inspect_local(self.repo, "v1.0.0", self.targets)
        remote = inspect_remote(self.repo, "origin", "v1.0.0")

        with self.assertRaises(PublishError) as caught:
            ensure_fast_forward(self.repo, "origin", identity, remote)

        self.assertEqual("not_started", caught.exception.status)
        self.assertIn("tag", str(caught.exception).lower())

    def test_rejects_unexpected_remote_url(self) -> None:
        git(self.repo, "remote", "set-url", "origin", str(self.root / "other.git"))

        with self.assertRaises(PublishError) as caught:
            inspect_local(self.repo, "v1.0.0", self.targets)

        self.assertEqual("not_started", caught.exception.status)
        self.assertIn("origin", str(caught.exception))


class PublishFlowTests(RepositoryCase):
    def test_dry_run_never_writes_either_remote(self) -> None:
        result = publish(
            self.repo,
            "v1.0.0",
            self.targets,
            execute=False,
            validate=False,
        )

        self.assertEqual("dry_run", result["status"])
        self.assertIsNone(inspect_remote(self.repo, "origin", "v1.0.0").main_commit)
        self.assertIsNone(inspect_remote(self.repo, "internal", "v1.0.0").main_commit)

    def test_execute_pushes_github_then_internal_and_verifies_both(self) -> None:
        result = publish(
            self.repo,
            "v1.0.0",
            self.targets,
            execute=True,
            validate=False,
        )
        identity = inspect_local(self.repo, "v1.0.0", self.targets)

        self.assertEqual("complete", result["status"])
        for remote_name in ("origin", "internal"):
            state = inspect_remote(self.repo, remote_name, "v1.0.0")
            self.assertEqual(identity.main_commit, state.main_commit)
            self.assertEqual(identity.tag_object, state.tag_object)
            self.assertEqual(identity.tag_commit, state.tag_commit)

    def test_github_failure_never_writes_internal(self) -> None:
        self.reject_pushes(self.github)

        with self.assertRaises(PublishError) as caught:
            publish(
                self.repo,
                "v1.0.0",
                self.targets,
                execute=True,
                validate=False,
            )

        self.assertEqual("github_failed", caught.exception.status)
        self.assertIsNone(inspect_remote(self.repo, "internal", "v1.0.0").main_commit)

    def test_internal_failure_reports_pending_after_github_succeeds(self) -> None:
        self.reject_pushes(self.internal)

        with self.assertRaises(PublishError) as caught:
            publish(
                self.repo,
                "v1.0.0",
                self.targets,
                execute=True,
                validate=False,
            )

        identity = inspect_local(self.repo, "v1.0.0", self.targets)
        github_state = inspect_remote(self.repo, "origin", "v1.0.0")
        self.assertEqual("internal_pending", caught.exception.status)
        self.assertEqual(identity.main_commit, github_state.main_commit)
        self.assertIsNone(inspect_remote(self.repo, "internal", "v1.0.0").main_commit)

    def test_cli_json_does_not_expose_url_credentials(self) -> None:
        git(
            self.repo,
            "remote",
            "set-url",
            "origin",
            "https://test-user:test-secret@example.invalid/repo.git",
        )
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            exit_code = main(
                ["--tag", "v1.0.0"],
                repo=self.repo,
                targets=self.targets,
                validate=False,
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(1, exit_code)
        self.assertEqual("not_started", payload["status"])
        self.assertNotIn("test-user", output.getvalue())
        self.assertNotIn("test-secret", output.getvalue())


if __name__ == "__main__":
    unittest.main()
