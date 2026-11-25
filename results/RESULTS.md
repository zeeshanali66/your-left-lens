# Does the model know YOUR left? Results

## Behaviour of five chat models (Ollama, one-word answer; chance 0.50)

| model | same | opposite | self_same | self_opp | answers 'flipped' when asked about the OTHER person |
|---|---|---|---|---|---|
| gemma4:12b | 0.01 [0.00, 0.04] | 0.97 [0.94, 1.00] | 0.99 [0.96, 1.00] | 0.75 [0.65, 0.84] | 0.98 |
| qwen3.5:9b | 0.38 [0.28, 0.49] | 0.96 [0.91, 1.00] | 1.00 [1.00, 1.00] | 0.94 [0.88, 0.99] | 0.79 |
| gemma3:12b | 0.30 [0.20, 0.40] | 0.89 [0.81, 0.95] | 0.79 [0.70, 0.88] | 0.72 [0.62, 0.82] | 0.79 |
| qwen2.5:7b | 0.09 [0.04, 0.15] | 0.97 [0.94, 1.00] | 1.00 [1.00, 1.00] | 0.99 [0.96, 1.00] | 0.94 |
| llama3.1:8b | 0.29 [0.19, 0.39] | 0.74 [0.64, 0.82] | 0.72 [0.62, 0.81] | 0.66 [0.55, 0.76] | 0.72 |

## qwen1.5b (28 layers)

### Behaviour (final layer accuracy, 95% bootstrap CI; chance 0.50)

| condition | accuracy | mean logit diff (correct - wrong) |
|---|---|---|
| same | 0.50 [0.39, 0.61] | -0.99 |
| opposite | 0.50 [0.39, 0.61] | +0.67 |
| self_same | 0.65 [0.55, 0.75] | +1.27 |
| self_opp | 0.72 [0.62, 0.82] | +1.74 |

### Logit lens: accuracy at each layer

```
layer       0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27
same     0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.42 0.49 0.50 0.53 0.50 0.50 0.66 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50
opposite 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.53 0.49 0.50 0.47 0.50 0.45 0.29 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50
self_same0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.35 0.50 0.50 0.44 0.50 0.50 0.75 0.50 0.50 0.53 0.54 0.62 0.65 0.59 0.64 0.65
self_opp 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.42 0.47 0.50 0.51 0.50 0.53 0.84 0.50 0.50 0.60 0.62 0.74 0.76 0.62 0.72 0.72
```

same items: best layer 18 (accuracy 0.66) vs final 0.50. Right at layer 18 but wrong at the output: 0.16.

opposite items: best layer 12 (accuracy 0.53) vs final 0.50. Right at layer 12 but wrong at the output: 0.26.

Effect of the relation sentence on the output (matched pairs): logit difference for the UNflipped side is -0.99 when facing the same way and -0.67 when facing each other (difference -0.32; 0 would mean the relation is ignored; share of pairs where facing the same way favours the unflipped side more: 0.34).

### Linear probe for the correct side (leave-one-template-out accuracy)

```
layer       0    2    4    6    8   10   12   14   16   18   20   22   24   26
same     0.62 0.50 0.50 0.50 0.66 0.53 0.50 0.66 0.69 0.65 0.59 0.50 0.50 0.50
opposite 0.30 0.33 0.50 0.50 0.35 0.47 0.49 0.42 0.47 0.34 0.38 0.59 0.51 0.50
self_same0.79 0.61 0.51 0.60 0.69 0.60 0.50 0.62 0.91 0.79 0.79 0.74 0.54 0.55
self_opp 0.61 0.65 0.59 0.50 0.76 0.59 0.57 0.53 0.69 0.78 0.70 0.69 0.50 0.56
```

### Patching the last-token output of each layer (normalised effect: 0 = no change, 1 = the run now behaves like the source)

```
layer             0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27
opp -> same   0.25  0.12  0.00  0.00  0.00  0.00  0.12  0.25  0.00 -0.12  0.12  0.12  0.25  0.00  0.25  0.38  0.12  0.25  0.38  0.38  0.50  0.50  0.75  0.75  1.00  1.00  1.12  1.00   (n=1)
same -> opp  -0.12  0.00  0.00  0.00  0.00  0.12  0.25  0.12  0.00  0.12  0.38  0.12  0.25  0.12  0.12  0.50  0.25  0.12  0.25  0.25  0.50  0.50  0.88  0.88  1.00  0.88  1.00  1.00   (n=1)
```
Pairs where the same-way and facing runs differ by less than 1 logit are skipped: if the relation does not change the output, there is nothing for patching to move. Reading: in early layers the last token has not yet read the relation, so patching it changes little (near 0); the curve rises to 1 at the layers where the relation's effect has arrived at the answer position.

### Fixes (templates 2-3 only; early-exit layer chosen on templates 0-1)

| method | same | opposite | self_same | self_opp | all |
|---|---|---|---|---|---|
| final layer (no fix) | 0.50 | 0.50 | 0.50 | 0.60 | 0.53 |
| early exit at layer 24 | 0.50 | 0.50 | 0.50 | 0.60 | 0.53 |
| step-by-step prompt | 0.20 | 0.65 | 0.53 | 0.55 | 0.48 |
| rule hint in the prompt | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |

## gemma4-e4b (42 layers)

### Behaviour (final layer accuracy, 95% bootstrap CI; chance 0.50)

| condition | accuracy | mean logit diff (correct - wrong) |
|---|---|---|
| same | 0.28 [0.19, 0.36] | -0.96 |
| opposite | 0.70 [0.60, 0.80] | +1.32 |
| self_same | 1.00 [1.00, 1.00] | +6.34 |
| self_opp | 1.00 [1.00, 1.00] | +6.10 |

### Logit lens: accuracy at each layer

```
layer       0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27   28   29   30   31   32   33   34   35   36   37   38   39   40   41
same     0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.55 0.50 0.53 0.50 0.50 0.50 0.50 0.50 0.50 0.56 0.62 0.54 0.51 0.57 0.59 0.30 0.28 0.25 0.30 0.30 0.36 0.34 0.40 0.28
opposite 0.50 0.50 0.47 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.53 0.50 0.50 0.49 0.50 0.57 0.50 0.50 0.50 0.50 0.50 0.50 0.41 0.47 0.39 0.54 0.45 0.55 0.72 0.68 0.61 0.55 0.56 0.68 0.54 0.59 0.70
self_same0.50 0.50 0.47 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.59 0.50 0.72 0.50 0.55 0.50 0.50 0.50 0.50 0.47 0.56 0.78 0.75 0.75 0.88 0.82 0.86 1.00 1.00 0.99 1.00 0.97 0.96 1.00
self_opp 0.50 0.50 0.45 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.75 0.50 0.50 0.50 0.50 0.50 0.50 0.56 0.54 0.88 0.80 0.75 0.88 0.89 0.91 1.00 0.99 1.00 0.95 1.00 1.00 1.00
```

same items: best layer 28 (accuracy 0.62) vs final 0.28. Right at layer 28 but wrong at the output: 0.47.

opposite items: best layer 33 (accuracy 0.72) vs final 0.70. Right at layer 33 but wrong at the output: 0.12.

Effect of the relation sentence on the output (matched pairs): logit difference for the UNflipped side is -0.96 when facing the same way and -1.32 when facing each other (difference +0.36; 0 would mean the relation is ignored; share of pairs where facing the same way favours the unflipped side more: 0.50).

### Linear probe for the correct side (leave-one-template-out accuracy)

```
layer       0    2    4    6    8   10   12   14   16   18   20   22   24   26   28   30   32   34   36   38   40
same     0.59 0.51 0.56 0.56 0.60 0.55 0.54 0.53 0.60 0.59 0.55 0.51 0.59 0.61 0.57 0.64 0.54 0.62 0.57 0.62 0.55
opposite 0.23 0.46 0.40 0.38 0.41 0.44 0.49 0.50 0.45 0.39 0.47 0.51 0.57 0.49 0.54 0.33 0.38 0.47 0.46 0.42 0.50
self_same0.56 0.56 0.59 0.61 0.70 0.56 0.50 0.69 0.68 0.74 0.74 0.74 0.97 0.91 0.88 0.93 0.97 0.88 0.88 0.91 0.90
self_opp 0.75 0.51 0.59 0.53 0.65 0.60 0.57 0.50 0.70 0.74 0.80 0.71 0.96 0.95 0.79 0.85 0.85 0.81 0.88 0.89 0.88
```

### Patching the last-token output of each layer (normalised effect: 0 = no change, 1 = the run now behaves like the source)

```
layer             0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27    28    29    30    31    32    33    34    35    36    37    38    39    40    41
opp -> same  n/a   (n=0)
same -> opp  n/a   (n=0)
```
Pairs where the same-way and facing runs differ by less than 1 logit are skipped: if the relation does not change the output, there is nothing for patching to move. Reading: in early layers the last token has not yet read the relation, so patching it changes little (near 0); the curve rises to 1 at the layers where the relation's effect has arrived at the answer position.

### Fixes (templates 2-3 only; early-exit layer chosen on templates 0-1)

| method | same | opposite | self_same | self_opp | all |
|---|---|---|---|---|---|
| final layer (no fix) | 0.30 | 0.57 | 1.00 | 1.00 | 0.72 |
| early exit at layer 41 | 0.30 | 0.57 | 1.00 | 1.00 | 0.72 |

