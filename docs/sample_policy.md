# Content Moderation Policy

## Copyright Escalation Rules

When a piece of submitted content is flagged with a copyright similarity score above 0.85,
it must be escalated to a human reviewer within 24 hours. Scores between 0.70 and 0.85
are queued for batch review on a 72-hour SLA. Anything below 0.70 is auto-approved with
an audit log entry.

## Confidence Thresholds

Automated moderation decisions require a model confidence of at least 0.90 to be acted on
without human review. Decisions in the 0.70 to 0.90 band are sent for human spot-check.
Below 0.70 the request is rejected with a generic error and logged for retraining.

## Auditability

Every moderation decision must persist:
- input hash (SHA-256)
- model version and prompt version
- top retrieved policy snippets
- final classifier label and confidence
- human reviewer ID if applicable

These records are retained for 18 months.
