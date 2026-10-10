# LM BabiStories (paper Figure 2, scaled budget)

| method | params | val loss (mean±std) | per-seed | time/run (min) |
|---|---|---|---|---|
| Biweight (1.5-entmax) (1L) | 45.7M | 2.162±0.016 | 1337:2.180, 1338:2.154, 1339:2.150 | 28 |
| Epanech. (sparsemax) (1L) | 45.7M | 2.203±0.091 | 1337:2.160, 1338:2.309, 1339:2.142 | 14 |
| Gaussian (softmax) (1L) | 45.7M | 2.175±0.026 | 1337:2.204, 1338:2.161, 1339:2.159 | 13 |
