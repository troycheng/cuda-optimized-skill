# Public Documentation and Brand Refresh Design

## Purpose

Present `cuda-kernel-optimizer` as a clear, credible open-source Codex skill
without weakening its evidence rules. A first-time reader should understand the
project, install it, choose the right workflow, and find detailed documentation
without reading the agent execution protocol or development history.

This refresh changes presentation and documentation only. It does not change
optimization behavior, evidence schemas, safety boundaries, test claims, or the
V2.5 version.

## Observed problems

- The README opens with an isolated 88 px icon and a long project description;
  the project name, mark, and value proposition do not form one visual identity.
- Installation and documentation links appear after protocol, acceptance, and
  compatibility detail instead of in the first screen.
- All ten README sections have similar visual weight, so readers cannot
  distinguish orientation from evidence detail.
- User documentation lives under the skill package while `docs/` is dominated
  by implementation plans and design history.
- The repository has no explicit user-document navigation layer comparable to
  a small documentation site.

## Design alternatives

### A. README-only refresh

Replace the hero and shorten both README files, leaving all other documentation
where it is. This is the smallest change, but it does not solve discoverability
or separate user documentation from internal history.

### B. Lightweight public documentation layer (selected)

Refresh the bilingual README files, add a horizontal wordmark, and introduce a
small MkDocs navigation layer over focused user pages. Keep `SKILL.md`, bundled
references, schemas, and historical Superpowers documents in place as separate
execution and development layers.

This provides the clearest improvement without copying the scale or maintenance
cost of the vLLM website. The user approved this direction after reviewing the
proposed information architecture.

### C. Full documentation portal

Add versioned documentation, internationalization, release notes, search
plugins, deployment automation, and a dedicated website. This would resemble a
large community project, but it is disproportionate to the current repository
and would create maintenance surfaces unrelated to optimization quality.

## Information layers

The repository will expose four distinct layers:

1. **Project landing page:** `README.md` and `README.zh-CN.md` explain identity,
   value, quick start, primary workflows, evidence principles, and links.
2. **User guide:** a small set of pages under `docs/` explains installation,
   workflow selection, evidence and safety, compatibility, and where to find
   reference material.
3. **Agent execution package:** `skills/cuda-kernel-optimizer/SKILL.md`, scripts,
   references, templates, examples, and schemas remain canonical for Codex.
4. **Development history:** `docs/superpowers/` remains available for audit and
   maintenance but is excluded from the public documentation navigation.

The new layer must summarize or route to canonical material instead of copying
large protocol sections. Detailed formal rules continue to live in the skill
package.

## Brand system

Retain the existing Thread Tile icon and Carbon + Cyan palette. The icon already
has a tested geometry and is recognizable at small sizes; replacing it would
discard useful continuity.

Add two horizontal SVG lockups:

- `asset/logo-wordmark.svg` for light backgrounds;
- `asset/logo-wordmark-dark.svg` for dark backgrounds.

Each lockup combines the Thread Tile symbol with the words `CUDA KERNEL` and
`OPTIMIZER`. The icon geometry and colors must match the existing light or dark
asset. Use a deterministic SVG font stack, transparent background, accessible
title, and no gradients, shadows, NVIDIA marks, or CUDA brand artwork.

The README hero uses the wordmark at a readable desktop width with a smaller
responsive maximum width, followed by this value proposition:

> Evidence-driven CUDA, CUTLASS and Triton optimization for Codex

The next line contains five high-value links: `Get Started`, `Workflows`,
`Evidence & Safety`, `Examples`, and the alternate README language.

The square SVG and PNG assets remain the application icon and are not removed.

## README structure

Both language versions use the same order and equivalent claims:

1. Centered wordmark, value proposition, and primary links.
2. **About:** one short definition and a compact list of differentiating
   capabilities.
3. **Quick start:** installation request for Codex, required inputs, and one
   minimal task prompt.
4. **Choose a workflow:** kernel loop, complete workload, serving validation,
   and read-only NCU analysis in one table.
5. **How it works:** one high-level diagram and a short explanation.
6. **Evidence, not best-sample claims:** correctness, paired measurements,
   frozen design, environment integrity, and fail-closed behavior.
7. **Tested scope:** current CPU/static and physical-GPU evidence with explicit
   boundaries.
8. **Documentation:** categorized user, agent, example, compatibility, and
   license links.

Move detailed budget tables, modification-scope tables, long deliverable lists,
and V2.5 protocol terminology out of the landing-page flow. Preserve their
meaning through concise boundaries and links. Installation must be visible
without scrolling through the evidence sections.

The README target is 120 to 170 lines per language. It may exceed that range
only when Markdown tables or the Mermaid diagram make the content clearer.

## Public documentation navigation

Add `mkdocs.yml` with project name, repository link, restrained Carbon + Cyan
colors, and this navigation:

- **Home** — `docs/index.md`
- **Getting Started** — `docs/getting-started.md`
- **Workflows** — `docs/workflows.md`
- **Evidence & Safety** — `docs/evidence-and-safety.md`
- **Compatibility** — `docs/compatibility.md`
- **Agent Protocol** — link to the repository `SKILL.md`

The public pages provide orientation and links rather than duplicate formal
schemas. `docs/superpowers/` is deliberately absent from the navigation. Add a
short `docs/superpowers/README.md` that identifies the directory as internal
design and implementation history.

Do not add versioning, deployment CI, analytics, comments, a blog, or duplicate
Chinese documentation in this change. The Chinese README remains the supported
Chinese entry point.

## Content boundaries

- Never present documented rules as automatically enforced unless a script or
  validator implements them.
- Keep `performance_verdict` separate from `evidence_integrity` in every public
  explanation of formal evidence.
- State that the installed self-check is CPU/static and does not validate a GPU
  environment.
- Preserve the distinction between kernel improvement and end-to-end workload
  improvement.
- Do not claim general speedups from repository fixtures or historical runs.
- Do not expose internal publishing commands or development-plan details in the
  public guide.

## Validation

Add or update CPU/static tests to verify:

- both README files use the correct light/dark wordmark assets and equivalent
  primary navigation;
- Quick Start precedes detailed evidence and compatibility sections;
- the wordmark SVG files parse as XML, use transparent backgrounds, contain the
  expected accessible titles, and reuse the established palette;
- `mkdocs.yml` contains the selected public navigation and excludes
  `docs/superpowers`;
- every public documentation page exists and all repository-relative Markdown
  links resolve;
- the V2.5 evidence-integrity boundary and self-check limitation remain present;
- the complete CPU/static test suite, skill validator, self-check, Markdown
  whitespace check, and repository diff check pass.

No GPU, NCU, remote host, `/data/triton-handoff`, or external-process validation
is required for this documentation-only change.

## Delivery scope

Expected implementation files are limited to:

- `README.md` and `README.zh-CN.md`;
- `asset/logo-wordmark.svg` and `asset/logo-wordmark-dark.svg`;
- `mkdocs.yml` and the five public pages under `docs/`;
- `docs/superpowers/README.md`;
- README, logo, and documentation-structure tests.

Runtime scripts, formal evidence schemas, installed skill contents, and GPU test
fixtures remain unchanged. The completed implementation should be reviewed and
tested locally before any push.
