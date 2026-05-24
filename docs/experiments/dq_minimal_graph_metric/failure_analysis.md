# d_q Failure Analysis

Failure-case comparison was not executed in this run.

The graph diagnostics gate failed before retrieval:

- HotpotQA disconnected-pair rate: `0.9642`
- 2WikiMultiHopQA disconnected-pair rate: `0.9625`

Because no graph-aware retrieval predictions were generated, there is no valid top-rho / semantic refine-cell / graph-aware refine-cell comparison yet. See `DQ_DECISION_REPORT.md` for the gating decision.
