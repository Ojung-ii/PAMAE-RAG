# Connectivity Backbone Gate Report

## Gate

A graph config passes the pre-retrieval gate if:

- `avg_disconnected_pair_rate <= 0.60`
- `avg_largest_component_ratio >= 0.50`
- `gold_support_connected_rate` improves over symbolic-only
- average degree is not excessive
- diagnostics are gold-free except the explicit support-connectivity evaluation metrics

## HotpotQA

| config | disconnected | connected | largest comp. | gold connected | avg degree | avg edges | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| symbolic_only | 0.9014 | 0.0986 | 0.2228 | 0.7895 | 4.55 | 1209.47 | fail |
| mutual_knn4 | 0.5102 | 0.4898 | 0.6711 | 0.9263 | 6.24 | 1657.98 | pass |
| mutual_knn8 | 0.1478 | 0.8522 | 0.9217 | 0.9579 | 8.23 | 2189.25 | pass, selected |
| knn4 | 0.0063 | 0.9937 | 0.9968 | 1.0000 | 9.58 | 2548.90 | pass, selected |
| knn8 | 0.0009 | 0.9991 | 0.9995 | 1.0000 | 14.38 | 3823.60 | pass, dense |

## 2WikiMultiHopQA

| config | disconnected | connected | largest comp. | gold connected | avg degree | avg edges | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| symbolic_only | 0.8770 | 0.1230 | 0.1769 | 0.5067 | 4.73 | 1230.17 | fail |
| mutual_knn4 | 0.5350 | 0.4650 | 0.6531 | 0.8133 | 6.48 | 1658.51 | pass |
| mutual_knn8 | 0.1357 | 0.8643 | 0.9282 | 0.9733 | 8.56 | 2167.88 | pass, selected |
| knn4 | 0.0079 | 0.9921 | 0.9959 | 1.0000 | 9.68 | 2454.84 | pass, selected |
| knn8 | 0.0014 | 0.9986 | 0.9993 | 1.0000 | 14.39 | 3616.07 | pass, dense |

## Selected Retrieval Settings

For each dataset, retrieval proceeds with:

- semantic refine-cell baseline
- top-rho baseline
- graph-sp refine-cell with `mutual_knn8`
- hybrid 0.7 semantic / 0.3 graph refine-cell with `mutual_knn8`
- graph-sp refine-cell with `knn4`
- hybrid 0.7 semantic / 0.3 graph refine-cell with `knn4`

`mutual_knn8` is the best sparse mutual-kNN setting. `knn4` is selected as the simple directed-kNN backbone because it essentially fixes connectivity while keeping average degree below 10; `knn8` is left out because it is substantially denser without much diagnostic gain.
