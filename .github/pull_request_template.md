<!--
Title: the outcome in plain words, as a sentence a non-expert can follow
("Read AV1 videos without dropping frames", not "fix(video): pyav backend").

Keep function names, flags and file paths out of everything above the folded
"Details for reviewers" block. Every number needs its unit and the conditions it
was measured under, and must come from a run, a file or the commit: never
estimated. Delete a section only if it truly doesn't apply; write "Nothing" or
"None" where that is the answer.
-->

## What this does, in plain words

<!-- Summary. The outcome first, the headline number in bold, then what it means
for someone using sleap-io. Link the issue or the PR this follows or replaces. -->

## Why

<!-- The problem: what went wrong, for whom, how it was found. Ideas that were
tried and didn't explain it or didn't work, each with its number. -->

## What changes

<!-- Key changes. One bold lead phrase per bullet: what it does, its default,
what it costs, how to turn it off. -->

## Measured

<!-- Conditions first (machine, data, versions, settings), then a before/after
table with median and p95/p99 where it applies, then a few bullets on what the
numbers mean. Say if runs were thrown away and why. -->

## What was checked

<!-- Testing: which tests, how many passed, on which Python and platform; what was
run by hand on real data; coverage of the changed lines. -->

## Not done / not verified

<!-- What wasn't tested or measured, why, and what would verify it. -->

## Limits

<!-- Known cases this doesn't handle or gets wrong. -->

## Design decisions

<!-- Alternatives considered and why they were rejected. What this deliberately
doesn't change. -->

## API changes and docs

<!-- New or changed public functions, arguments or defaults, or "None". What
changed in docs/, or why nothing there needed to. -->

<details>
<summary>Details for reviewers</summary>

<!-- Files and functions changed, example usage as code, internals that matter,
detailed tables, unrelated flaky tests seen while testing. -->

</details>
