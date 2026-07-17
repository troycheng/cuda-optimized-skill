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


if __name__ == "__main__":
    unittest.main()
