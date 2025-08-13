## ELFS: Entropy-based Loss Function Selection for Global Model Accuracy in Federated Learning (built on FL-bench)
Official implementation of **ELFS (Entropy-based Loss Function Selection)** for Federated Learning (FL), including code, configs, and all paper experiments.
Built on top of FL-bench for reproducibility.
<br/><br/><br/>

### TL;DR
* **What**: Dynamically selects the loss function (e.g., Cross-Entropy, Focal) per round/aggregation step **based on entropy signals**, improving **global model accuracy** under Non-IID client distributions.
* **Why**: Different loss functions excel under different data skews/hardness; ELFS chooses the right one when it matters.
* **How**: Compute entropy statistics → compare to threshold/schedule → pick the loss from a candidate pool → train and aggregate.
<br/>

### Environment
* Python 3.8+
* PyTorch (CUDA optional but recommended)
* Install dependencies:

```
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Adjust PyTorch/CUDA to your system if needed.
<br/>


### Quickstart

    # 1) Generate Non-IID data 
    python generate_data.py -d mnist -a 0.1 -cn 100

    # 2) Train (elfs)
    python main.py method=fedavg
> That’s it—these two commands reproduce the basic experiment.
<br/>

### Acknowledgement
This work builds on FL-bench: https://github.com/KarhouTam/FL-bench


### License
Choose a license for this repository (e.g., MIT/Apache-2.0) and ensure FL-bench’s license terms are followed and attributed.
