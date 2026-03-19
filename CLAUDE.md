# BT4221 HDB Resale Price Prediction

## Project Context

This is a PySpark-based ML project for predicting HDB resale prices in Singapore. User stories are tracked as GitHub Issues and organized on a GitHub Project board.

## User Story Search

When the user asks to search for, find, list, or show **user stories** (or issues, tasks, tickets), always fetch them live from GitHub using the `gh` CLI. Do NOT guess or rely on memory — always query GitHub directly.

### How to search

**Browse the full project board** (shows status: Todo / In Progress / Done):

```sh
gh project item-list 4 --owner yongggquannn --format json
```

**List issues with optional filters:**

```sh
# All open issues
gh issue list --state open

# Filter by sprint label
gh issue list --label "Sprint 1"
gh issue list --label "Sprint 2"

# Search by keyword in title/body
gh issue list --search "geocode"
gh issue list --search "model"

# Show closed/completed issues
gh issue list --state closed
```

**View full details of a specific user story** (includes acceptance criteria, assignees, body):

```sh
gh issue view <number>
```

### Output format

When presenting user stories, display a summary table with these columns:

| # | Title | Status | Assignee | Sprint |
|---|-------|--------|----------|--------|

Then ask the user if they want to see full details (acceptance criteria) for any specific user story.

### Matching user intent

Treat any of these phrases as a request to search GitHub Issues/Projects:
- "search for user stories"
- "find user stories"
- "show user stories"
- "list user stories"
- "what are the tasks"
- "show me the backlog"
- "what's in sprint X"
- "show issues"
- "what needs to be done"
- "check the project board"
