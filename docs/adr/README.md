# Architecture Decision Records

Short records of the decisions that shape this project, in the order they were made.
Each one states the context at the time, what was decided, and what it costs.
A decision is never edited after acceptance; if it changes, a new record supersedes it.

| # | Decision | Status |
|---|---|---|
| [0001](0001-project-tooling.md) | Python project layout and tooling | Accepted |
| [0002](0002-exclude-user-identifiers.md) | Exclude third-party user identifiers from all derived data | Accepted |
| [0003](0003-target-definition.md) | Predict log midpoint and log spread of the USD range | Accepted |
| [0004](0004-drop-not-repair-implausible-pay.md) | Drop implausible compensation rows rather than repair them | Accepted |
| [0005](0005-scrub-salary-mentions.md) | Scrub salary figures from text before featurisation | Accepted |
| [0006](0006-split-protocol.md) | Group splits by collapse key, hold out the newest postings | Accepted |
| [0007](0007-maskable-feature-blocks.md) | Support masked inputs with grouped feature blocks and block dropout | Proposed |
| [0008](0008-two-deployment-targets.md) | Ship a browser-only demo and a separate service deployment | Proposed |

Template: [template.md](template.md).
