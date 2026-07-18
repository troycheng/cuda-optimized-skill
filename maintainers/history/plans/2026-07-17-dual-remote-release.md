# GitHub 与内网 GitLab 受控双发布实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 增加一个默认 dry-run 的本地发布工具，只把已验证的 `main` 和指定版本标签依次发布到用户的 GitHub fork 与内网 GitLab，并完成现有 `v2.3.0` 的首次内网同步。

**架构：** 发布工具把 Git 操作封装为不经过 shell 的参数数组。可注入的 `ReleaseTargets` 负责远端身份，可测试的 preflight 负责本地身份与远端拓扑，publisher 固定按 GitHub、GitLab 顺序执行原子双 ref push，并在每一步回读。CLI 使用生产目标和完整测试；单元测试使用临时 bare repository，不访问真实远端。

**技术栈：** Python 3 标准库、Git CLI、`unittest`、临时 bare repository。

---

## 文件结构

- 创建 `tools/__init__.py`：让测试可直接导入发布模块。
- 创建 `tools/publish_dual_remote.py`：远端身份、本地检查、拓扑判断、发布和 JSON 结果。
- 创建 `tests/test_publish_dual_remote.py`：临时仓库单元与集成测试。
- 修改 `README.md`：说明权威仓库、内网镜像和发布入口。
- 修改 `README.zh-CN.md`：独立写出相同事实，不逐句翻译。

### 任务 1：实现本地身份与远端拓扑检查

**文件：**
- 创建：`tools/__init__.py`
- 创建：`tools/publish_dual_remote.py`
- 创建：`tests/test_publish_dual_remote.py`

- [x] **步骤 1：为本地身份和拓扑规则编写失败测试**

测试用 `tempfile.TemporaryDirectory()` 创建工作仓库和两个 bare remote，辅助函数只运行
参数数组：

```python
def git(cwd, *args, check=True):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
```

`ReleaseTargets` 在测试中指向临时 bare repository；生产常量仍使用规格中的精确 URL。
`PreflightTests` 必须包含以下完整测试：

- `test_rejects_dirty_tree_and_lightweight_tag`：先写未跟踪文件，再创建轻量标签，分别断言
  `PublishError.status == "not_started"`。
- `test_accepts_empty_remote_and_matching_annotated_tag`：创建带注释标签，断言空远端状态允许
  首次发布。
- `test_rejects_diverged_remote_main`：在 bare remote 的独立 clone 创建旁支提交，断言拒绝。
- `test_rejects_conflicting_remote_tag`：让同名标签指向另一个 commit，断言拒绝。
- `test_repeated_matching_refs_are_idempotent`：远端 refs 与本地完全一致，断言检查成功且不要求
  写入。

- [x] **步骤 2：运行测试，确认 RED**

运行：

```bash
python3 -m unittest tests.test_publish_dual_remote.PreflightTests -v
```

预期：FAIL，`tools.publish_dual_remote` 尚不存在。

- [x] **步骤 3：实现最小 preflight**

模块先实现以下稳定数据类型：

```python
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
```

同时定义 `PublishError(message: str, status: str)`、
`inspect_local(repo: Path, tag: str, targets: ReleaseTargets) -> ReleaseIdentity`、
`inspect_remote(repo: Path, remote: str, tag: str) -> RemoteState` 和
`ensure_fast_forward(repo: Path, remote: str, local: ReleaseIdentity,
remote_state: RemoteState) -> None`。

`inspect_local` 检查当前分支、工作区、三个 remote URL、`upstream` push URL、带注释标签和
标签可达性。`inspect_remote` 使用 `ls-remote` 读取精确 refs；远端存在 `main` 时 fetch 到
独立临时 ref，再由 `merge-base --is-ancestor` 判断。任何不一致都抛出
`PublishError(status="not_started")`。

- [x] **步骤 4：运行测试，确认 GREEN**

运行：

```bash
python3 -m unittest tests.test_publish_dual_remote.PreflightTests -v
```

预期：全部通过。

- [x] **步骤 5：提交核心检查**

```bash
git add tools/__init__.py tools/publish_dual_remote.py tests/test_publish_dual_remote.py
git commit -m "feat(发布): 添加双远端安全检查"
```

### 任务 2：实现 dry-run、顺序发布与失败状态

**文件：**
- 修改：`tools/publish_dual_remote.py`
- 修改：`tests/test_publish_dual_remote.py`
- 修改：`README.md`
- 修改：`README.zh-CN.md`

- [x] **步骤 1：为发布顺序和失败语义编写失败测试**

`PublishFlowTests` 必须包含 5 个完整测试：

- `test_dry_run_never_writes_remote`：保存两个远端的 refs，调用 dry-run，再断言 refs 不变。
- `test_execute_pushes_github_then_internal_and_verifies_both`：执行发布后比较两个远端的 3 项
  身份。
- `test_github_failure_never_writes_internal`：注入在第一次 push 失败的 runner，断言 GitLab
  refs 为空。
- `test_internal_failure_reports_internal_pending`：注入在第二次 push 失败的 runner，断言
  GitHub 已更新且状态为 `internal_pending`。
- `test_cli_json_contains_no_url_credentials`：使用带测试 userinfo 的 URL 触发错误，断言
  JSON 中没有 userinfo 内容。

成功测试检查两个 bare remote 的 `refs/heads/main`、标签对象和 peeled commit；失败测试使用
只读目录或注入的 Git runner 让指定 push 失败，并检查调用顺序。

- [x] **步骤 2：运行测试，确认 RED**

运行：

```bash
python3 -m unittest tests.test_publish_dual_remote.PublishFlowTests -v
```

预期：FAIL，尚无 publisher 和 CLI。

- [x] **步骤 3：实现最小发布流程和 CLI**

实现 `publish(repo: Path, tag: str, targets: ReleaseTargets, *, execute: bool,
validate: bool = True) -> dict[str, object]`。参数 `validate=False` 只供 Python 单元测试注入，
CLI 不暴露该开关。CLI parser 使用以下固定定义：

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--execute", action="store_true")
    return parser
```

生产 CLI 在 preflight 后运行完整 CPU 测试，再按固定顺序执行：

```text
git push --atomic <remote> \
  refs/heads/main:refs/heads/main \
  refs/tags/<tag>:refs/tags/<tag>
```

每次 push 后重新 `ls-remote` 并比较三项身份。GitHub 失败返回 `github_failed`，不调用
GitLab；GitLab 失败返回 `internal_pending`。默认 dry-run 只输出 `not_started` 之外的
计划状态，不执行 push。测试可从 Python 接口传入 `validate=False`，CLI 不提供跳过测试的
参数。

- [x] **步骤 4：更新双语 README**

在两份 README 的兼容性与验证部分增加简短的「仓库发布」说明：GitHub 是权威源，GitLab
是单向镜像，只同步 `main` 和发布标签；给出 dry-run 和 `--execute` 命令；明确禁止从
GitLab 反向开发和覆盖 GitHub。

- [x] **步骤 5：运行 focused tests 和文档同步测试**

运行：

```bash
python3 -m unittest \
  tests.test_publish_dual_remote \
  tests.test_readme_sync -v
python3 tools/publish_dual_remote.py --help
git diff --check
```

预期：全部通过，help 退出码为 0，diff 无格式错误。

- [x] **步骤 6：提交发布流程**

```bash
git add tools/publish_dual_remote.py tests/test_publish_dual_remote.py README.md README.zh-CN.md
git commit -m "feat(发布): 支持 GitHub 与 GitLab 受控双发布"
```

### 任务 3：完整验证、首次同步和集成

**文件：**
- 修改：仅在验证发现真实缺陷时修改任务 1 或任务 2 已列文件。

- [x] **步骤 1：运行完整本地验证**

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -q
python3 -m compileall -q tools skills/cuda-kernel-optimizer/scripts tests
python3 /Users/tcheng/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  skills/cuda-kernel-optimizer
git diff --check
git status --short --branch
```

预期：全部测试通过，skill 有效，工作区只包含计划执行记录或为空。

- [x] **步骤 2：配置并只读检查内网 remote**

```bash
git remote add internal git@git.yukework.com:mlsys/cuda-optimized-skill.git
git remote get-url internal
GIT_TERMINAL_PROMPT=0 git ls-remote internal \
  refs/heads/main refs/tags/v2.3.0 'refs/tags/v2.3.0^{}'
```

如果 `internal` 已存在，必须先确认 URL 完全一致，不能覆盖现有配置。

- [x] **步骤 3：首次同步既有正式 refs**

从主 checkout 的干净 `main` 执行 dry-run 检查后，只把已经发布的 refs 推到内网：

```bash
git push --atomic internal \
  refs/remotes/origin/main:refs/heads/main \
  refs/tags/v2.3.0:refs/tags/v2.3.0
```

不得推送 `agent/dual-remote-release` 或其他开发分支。

- [x] **步骤 4：回读两个平台并比较**

```bash
git ls-remote origin refs/heads/main refs/tags/v2.3.0 'refs/tags/v2.3.0^{}'
git ls-remote internal refs/heads/main refs/tags/v2.3.0 'refs/tags/v2.3.0^{}'
```

预期：两边三项 ref 完全一致，`main` 和 peeled tag commit 都是
`416f416fe37a3834a92c8849c3fe7dd79c8a7c3a`。

- [x] **步骤 5：更新计划执行记录并提交**

把实测测试数、首次同步 commit/tag 和两个远端回读结果写到本计划末尾，再提交：

```bash
git add docs/superpowers/plans/2026-07-17-dual-remote-release.md
git commit -m "docs(发布): 记录双远端验证结果"
```

- [x] **步骤 6：合并并发布工具**

确认 feature branch 只包含设计、计划、工具、测试和双语 README。快进合并到本地 `main`，
在 `main` 重跑 focused tests，然后使用新工具把更新后的 `main` 依次推到 GitHub 与 GitLab；
`v2.3.0` 标签保持不变。最终回读两个远端 `main`，不得向 `upstream` 推送。

### 执行记录（2026-07-17）

- 新增测试进入完整回归后共运行 623 项：619 项通过，4 项 opt-in GPU 测试跳过，
  失败为 0，用时 52.521 秒。
- `compileall`、skill validator、CLI help 和 `git diff --check` 全部通过。
- 快进合并后的主干 focused tests 共 29 项，全部通过。
- `internal` 使用 SSH 地址 `git@git.yukework.com:mlsys/cuda-optimized-skill.git`；
  `upstream` push URL 保持 `DISABLED`。
- 首次同步只创建内网 `main` 和 `v2.3.0`，没有推送任何开发分支。
- GitHub 与内网 GitLab 回读结果完全一致：`main` 和 peeled tag commit 均为
  `416f416fe37a3834a92c8849c3fe7dd79c8a7c3a`，annotated tag object 均为
  `d64835607ee86bdd29cb56b37616197def73cdd2`。
