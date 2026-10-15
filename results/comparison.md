| Model | Seeds | Test macro-AUROC | Test macro-F1 | Val macro-AUROC | Params | ms/ECG |
|---|---|---|---|---|---|---|
| CNN (1D ResNet) | 3 | 0.9201 ± 0.0007 | 0.7386 ± 0.0031 | 0.9261 | 957,189 | 0.9 |
| LSTM (BiLSTM) | 3 | 0.9199 ± 0.0016 | 0.7374 ± 0.0045 | 0.9250 | 600,966 | 4.5 |
| Transformer | 3 | 0.9011 ± 0.0009 | 0.7091 ± 0.0071 | 0.9090 | 559,365 | 0.9 |
| Late-fusion Ensemble | 3 | 0.9265 ± 0.0003 | 0.7487 ± 0.0057 | 0.9319 | 2,117,520 | nan |
| Parallel Fusion (CNN+LSTM+Transformer) | 3 | 0.9223 ± 0.0019 | 0.7423 ± 0.0051 | 0.9257 | 2,211,621 | 6.5 |

Reference only: Strodthoff et al. (2021, IEEE JBHI) report macro-AUROC around 0.93 on the same superdiagnostic task. Those are their numbers, not ours.
