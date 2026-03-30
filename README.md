# Neural Corrective Machine Unranking (CuRD)

This repository provides the official implementation of the paper:

**"Neural Corrective Machine Unranking"**  
*Information Sciences, 2026*

The paper introduces a formal definition of **corrective machine unlearning** in neural information retrieval and proposes a novel method:

> **Corrective Unranking Distillation (CuRD)**

---

## 📌 Overview

Machine unlearning in information retrieval aims to remove the influence of specific data from trained ranking models.

In this work, we:

- Define **corrective unranking** for neural IR
- Provide a unified experimental framework
- Reproduce multiple machine unlearning baselines
- Propose a new method: **CuRD**
- Include an additional loss design: **CoCoL (Contrastive and Consistent Loss)**

---

## 📁 Repository Structure

```
.
├── Conf/                  # Configuration files for datasets and models
├── Data/                  # Datasets
├── models/                # Neural ranking model implementations
├── models_saved/          # Pretrained model checkpoints
├── results/               # Prediction outputs
│
├── ranking_dataset.py     # Dataset construction
├── task_utils.py          # Utility functions
├── unranking_methods.py   # Machine unlearning methods
├── unranking_task.py      # Unranking workflows
├── train.py               # Training / backprop / gradient control
├── task_launcher.py       # Task configuration & entry point
├── task_eval.py           # Evaluation scripts
│
└── ssd.py                 # SSD implementation (external source)
```

---

## 🧠 Supported Models

- BERTcat  
- BERTdot  
- ColBERT  
- PARADE  

In addition, the repository also includes implementations of traditional embedding-based IR models such as **DRMM, KNRM, DUET, and ConvKNRM**, although they are not used in our experiments.

CuRD is generally applicable to neural information retrieval methods following the  **representation + ranking** paradigm.

## 📊 Datasets

- MS MARCO
- TREC CAR (Year 3 & Year 4)

We also provide:

- Corrective unranking datasets  
- Pretrained ranking models  

---

## 🔬 Implemented Methods

We reproduce **7 machine unlearning baselines**:

- amnesiac
- retrain
- catastrophic
- NegGrad
- BadTeacher
- SSD
- CoCoL (Contrastive and Consistent Loss)

Proposed method:

- CuRD (Corrective Unranking Distillation)

---

## 📥 Data & Checkpoints

https://1drv.ms/f/c/00c07038f4fdc681/IgC_c3qKUQxvS7Z3rbjxwDf0AWpElzh7agp6QIvqvwBVLYE

---

## 🚀 Quick Start

### 1. Configure the task
Edit files in:
```
Conf/
```

### 2. Launch a task
```
python task_launcher.py
```

### 3. Evaluate results
```
python task_eval.py
```

---

## 📎 External Resources

SSD implementation is adapted from:
https://github.com/if-loops/selective-synaptic-dampening

---

## 📬 Contact

jhou.research@outlook.com

---

## 📖 Citation

```bibtex
@article{HOU2026123366,
  title   = {Neural corrective machine unranking},
  journal = {Information Sciences},
  volume  = {745},
  pages   = {123366},
  year    = {2026},
  issn    = {0020-0255},
  doi     = {https://doi.org/10.1016/j.ins.2026.123366},
  url     = {https://www.sciencedirect.com/science/article/pii/S0020025526002975},
  author  = {Jingrui Hou and Axel Finke and Georgina Cosma}
}
```

---

## 📄 License

This project is released for research purposes only.

Please consult the licenses of the corresponding datasets for usage restrictions.
