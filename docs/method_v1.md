# PAMAE-RAG v1 method note

## Scope

This repository starts with a strict PAMAE-consistent baseline. The terminal-conditioned evidence-chain idea is not included in v1. It should be added only after v1 shows measurable behavior on retrieval diagnostics.

## Mapping from PAMAE to GraphRAG

| PAMAE | PAMAE-RAG |
|---|---|
| entire data `D` | query-relevant universe `V_q` |
| sample `S_r` | query-sampled anchor candidate subset |
| medoid set `Θ` | anchor set `A` |
| clustering error `φ(Θ)` | query anchor distortion `L_q(A)` |
| Phase I global search | exact sample-level anchor search |
| Step 3 selection on entire data | full `V_q` objective validation |
| Phase II refinement | full `V_q` partition and monotone medoid update |

## Objective

```latex
\mathcal{L}_q(A)
=
\sum_{v\in V_q}\rho_q(v)\min_{a\in A}d_q(v,a)
+
\lambda_T\sum_{a\in A}T(a)
+
\lambda_k|A|.
```

The same objective must be used in all three locations: sample-level search, full-universe seed selection, and local refinement acceptance.

## Distance

The v1 default is angular embedding distance:

```latex
 d_{ang}(u,v) = \arccos(\langle \tilde{x}_u, \tilde{x}_v\rangle)/\pi.
```

Graph shortest-path distance should be added only after query graph edge lengths are non-negative and stable.

## Deferred v2 extension

Terminal-conditioned posterior will replace or augment relevance mass:

```latex
\rho_q^{term}(v)
\propto
\sum_{s,t}
\mu_q(s)\eta_q(t)
\exp\{-\beta[d_q(s,v)+d_q(v,t)-d_q(s,t)]\}
r_q(v).
```
