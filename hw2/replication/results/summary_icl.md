# RegBench ICL (paper Table 1)

| method | acc@1k | acc@5k | tvd@1k | tvd@5k |
|---|---|---|---|---|
| Biweight (1.5-entmax) | 0.844 | 0.875 | 119.5 | 62.1 |
| Epanech. (sparsemax) | 0.853 | 0.875 | 118.7 | 61.8 |
| Epanech. (normReLU) | 0.802 | 0.808 | 124.7 | 68.6 |
| Epanech. (1-ReLUmax) | 0.851 | 0.880 | 121.2 | 60.4 |
| Gaussian (softmax) | 0.850 | 0.875 | 119.4 | 62.4 |
| Gaussian kNN k=16 | 0.849 | 0.864 | 118.6 | 59.1 |
| Uniform kNN k=16 | 0.769 | 0.770 | 127.1 | 75.9 |
| Triweight (4/3-entmax) | 0.842 | 0.868 | 119.4 | 62.8 |
