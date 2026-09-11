# External integrations

This document describes how to add and manage external GitHub repositories that
should be checked out or optionally installed by CI for integration testing or
runtime use.

How it works
------------
- Add one external repo per line to `extern_repos.txt` in either of the forms:
  - owner/repo
  - https://github.com/owner/repo.git@<commit-or-tag>

- When `extern_repos.txt` is present and non-empty, CI will clone each listed
  repository into `extern/<owner>_<repo>` during the CI run.

- By default CI only clones the repositories. If you want CI to `pip install`
  or run the external repo's tests, edit `tools/checkout_externals.sh` and
  uncomment or add the relevant commands (the script contains examples).

Reproducibility
---------------
- For reproducible builds and tests prefer specifying a commit SHA or tag in the
  extern_repos.txt entry: ``https://github.com/OWNER/REPO.git@<commit-sha>``
- Alternatively, add pip-style git dependencies to `requirements.txt`:
  ``git+https://github.com/OWNER/REPO.git@<commit-sha>#egg=repo_name``

CI behavior and safety
----------------------
- The CI step that processes extern_repos.txt is a no-op if the file is absent
  or empty.
- CI will not automatically run installs or tests for external repos unless
  you enable those lines in `tools/checkout_externals.sh`.

Local usage
-----------
- To test locally, clone this repo, create a branch, add the desired repos to
  `extern_repos.txt`, and push the branch. The CI pipeline will pick them up on
  the next run.

Examples
--------
# Simple owner/repo entry (will use latest default branch):
numpy/numpy

# Example pinned to a commit (recommended for CI reproducibility):
# https://github.com/OWNER/REPO.git@abcdef1234567890abcdef1234567890abcdef12
