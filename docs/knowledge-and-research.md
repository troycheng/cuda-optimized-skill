# Knowledge, search, and independent challenge

The optimizer uses three evidence layers:

1. **Local facts** — source, environment probes, compiler output, profiler data,
   raw benchmark samples, and workload KPIs. These decide whether a change wins.
2. **Bundled knowledge** — architecture capabilities, method cards, profiler
   guidance, workload bottlenecks, compatibility notes, and a dated source
   manifest. This keeps the skill useful without internet access.
3. **External research** — current primary documentation and optional independent
   model critiques. This broadens ideas but remains advisory.

## Offline operation

The repository carries compact, machine-queryable knowledge rather than copies
of entire vendor manuals. Each source records a version or documentation
snapshot, verification date, and staleness policy. Unknown architectures and
version mismatches fail closed: the agent must probe locally or report that the
fact is unverified.

The Capability Registry returns only a few cards matching the exact
architecture, optimization layer, observed signal set, and available evidence.
Each card names counter-signals, required gates, verification steps, and stop
conditions. A hard UTF-8 byte budget limits the loaded playbooks; the complete
catalog never enters the Planner context.

During V3.1 active diagnosis, a smaller routing catalog supplies at most three
diagnostic cards for the current layer and bottleneck. The cards name likely signals,
counter-signals, and the next evidence family; they contain no commands and have
`promotion_authority: none`. They help the model read the context without loading the
full knowledge base and cannot override an observed outcome.

Registry matches propose methods. They cannot admit an experiment or promote a
candidate. The Controller re-reads sealed local artifacts and decides whether
the required gates are satisfied.

## External search

When network access and policy allow it, the agent searches primary vendor
documentation, source repositories, specifications, and papers for version-
specific questions. It records the source and the local observation the source
is intended to explain.

## Independent model challenge

External models are most useful for major direction choices, unexplained
plateaus, repeated failure of one mechanism, or review of a new compatibility
claim. They receive a small, redacted evidence packet and answer independently
before seeing other proposals. Their assumptions and proposed falsification
tests are recorded; disagreement is preserved rather than converted into a
vote.

Private source, credentials, raw logs, inputs, and hostnames are not sent
outside the environment without explicit approval. External systems never run
the target, modify the repository, change the host, or promote a candidate.

If search or model providers are unavailable, the local workflow continues.
