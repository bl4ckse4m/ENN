# RegBench ICL (paper Table 1)

| method | acc@1k | acc@5k | tvd@1k | tvd@5k |
|---|---|---|---|---|
| Biweight (1.5-entmax) | 0.857 | 0.875 | 66.1 | 62.1 |
| Epanech. (sparsemax) | 0.863 | 0.875 | 64.7 | 61.8 |
| Epanech. (normReLU) | 0.806 | 0.808 | 69.6 | 68.6 |
| Epanech. (1-ReLUmax) | 0.865 | 0.880 | 67.0 | 60.4 |
| Gaussian (softmax) | 0.861 | 0.875 | 65.4 | 62.4 |
| Gaussian kNN k=16 | 0.843 | 0.864 | 65.7 | 59.1 |
| Uniform kNN k=16 | 0.764 | 0.770 | 78.7 | 75.9 |
| Triweight (4/3-entmax) | 0.858 | 0.868 | 64.4 | 62.8 |

lr: 1k=0.0005, 5k=0.0001
