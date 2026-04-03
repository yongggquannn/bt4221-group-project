## Notebook Comments

Never reference user story IDs (e.g. `US-2.1`, `US-5.4`) in code comments, markdown cells, or print statements inside the notebook. Use plain descriptive labels instead (e.g. "Distance features", "School quality features"). User story IDs belong in GitHub issues and commit messages, not in the codebase.

## User Story Search

For **user stories**, **issues**, **tasks**, **tickets**, backlog, sprints, or “what needs to be done”: query GitHub with **`gh`** — never guess.

**Commands**

```sh
gh project item-list 4 --owner yongggquannn --format json   # board: Todo / In Progress / Done
gh issue list --state open                                   # add: --label "Sprint N" | --search "kw" | --state closed
gh issue view <number>                                       # full body, assignees, acceptance criteria
```

**Present** a table `| # | Title | Status | Assignee | Sprint |`, then offer full details (`gh issue view`) for any issue they pick.

**Triggers** (same intent): e.g. list/show/find user stories, tasks, backlog, sprint work, project board, open issues.
