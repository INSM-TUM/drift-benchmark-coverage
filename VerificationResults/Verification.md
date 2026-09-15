## Results and Interpretation

### CDLG and CDRIFT Benchmark Performance

Table 1 reports change point detection accuracy across the 7,500 test logs of the CDLG benchmark and the 115 logs of the independent CDRIFT benchmark suite.

**Table 1:** Detection Accuracy on Synthetic Benchmark Suites across Latencies $\gamma$.

| Benchmark | Representation | Latency ($\gamma$) | Precision | Recall | $F_1$-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **CDLG** ($N=200$) | `ARM` (Ours) | 1.0% | 0.842 | 0.735 | 0.784 |
| | `ARM` (Ours) | 2.5% | 0.873 | 0.762 | 0.814 |
| | `ARM` (Ours) | 5.0% | 0.877 | 0.765 | 0.817 |
| | `DFG` (Baseline) | 1.0% | 0.873 | 0.747 | 0.805 |
| | `DFG` (Baseline) | 2.5% | 0.902 | 0.771 | 0.831 |
| | `DFG` (Baseline) | 5.0% | 0.905 | 0.774 | 0.834 |
| **CDRIFT** ($N=70$) | `ARM` (Ours) | 1.0% | 0.642 | 0.548 | 0.591 |
| | `ARM` (Ours) | 2.5% | 0.993 | 0.847 | 0.914 |
| | `ARM` (Ours) | 5.0% | 0.993 | 0.847 | 0.914 |
| | `DFG` (Baseline) | 1.0% | 0.607 | 0.522 | 0.562 |
| | `DFG` (Baseline) | 2.5% | 1.000 | 0.860 | 0.925 |
| | `DFG` (Baseline) | 5.0% | 1.000 | 0.860 | 0.925 |

On both benchmark suites, all representations perform near-identically. At $\gamma = 5.0\%$, DFG reaches an $F_1$-score of $0.834$ on CDLG and $0.925$ on CDRIFT, while ARM achieves $0.817$ and $0.914$, respectively. Under extreme trace noise on CDLG (30% and 60% noisy traces), ARM maintains stable detection scores ($F_1 = 0.791$ and $F_1 = 0.800$), confirming that the continuous 8D formulation prevents discrete state instability under noise.

**Why DFG Succeeds on Existing Benchmarks:**  
The near-parity in accuracy on CDLG and CDRIFT is an artifact of benchmark construction rather than representational sufficiency. Because CDLG mutates imperative process trees and CDRIFT exercises structural change patterns, any change introduced to branching or activity execution indirectly alters the local directly-follows transition probabilities of surrounding activities as shown in the setup section. These secondary transition effects allow DFG-based detectors to pick up structural modifications without explicitly modeling existential constraints.