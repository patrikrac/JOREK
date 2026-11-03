# Contributing to JOREK

Thanks for your interest in contributing to JOREK. For more information, please refer to: 
<https://lukethewalker.github.io/doc_repo/docs/code_development/development_workflow.html>

## Repository model

Development uses a central main repository together with per-developer forks.
You work in your own fork and propose changes back through pull requests.

## Branches

- Features: `feature/<branch-name>`
- Bugfixes: `bugfix/<branch-name>`

Keep branches small and focused, develop bugfixes independently from features,
and sync with upstream regularly so that merge conflicts stay manageable.

## Pull requests

- Target the `master` branch of the main repository.
- The automatic regression tests must pass.
- At least two people must approve before a merge.
- Reference issues as `Fixes #<issue-number>` so they close automatically.
- Every contribution needs a Developer Certificate of Origin sign-off
  (`Signed-off-by: Name <email>`), as required by [GOVERNANCE.md](GOVERNANCE.md).

## Tests and coding standards

Create new regression tests for your developments — they are expected for all
contributions. Follow the coding guidelines in the developer documentation
before submitting.

## Issues

Report bugs and request features through the GitHub issue tracker. Give a clear,
concise description with steps to reproduce, and apply the appropriate label
(`bug`, `enhancement`, `discussion`).

## Publication rules

Publications and conference contributions must follow the
[publication rules](docs/publication_rules.md).
