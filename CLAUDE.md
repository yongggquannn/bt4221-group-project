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
