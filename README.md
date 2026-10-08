# SkimLit-NLP-Classifier-r 📄🔬

> A Sequential Sentence Classification deep NLP pipeline replicating *"PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts"* (Dernoncourt & Lee, 2017) using TensorFlow and multi-modal neural architectures.

## 📌 Overview
**SkimLit-NLP-Classifier-r** automates medical literature screening by classifying unstructured sentences in clinical trial abstracts into their corresponding structural roles:
- `BACKGROUND`
- `OBJECTIVE`
- `METHODS`
- `RESULTS`
- `CONCLUSIONS`

Instead of relying solely on simple word bags, this project constructs a **hybrid / multi-input deep neural network** in the TensorFlow Functional API that fuses semantic word tokens, sub-word character features, and document-level positional dynamics.

## 🧠 Model Architecture & Multi-Modal Fusion
```text
  Line Numbers (15-dim)    Total Lines (20-dim)      Token Sequence           Character Sequence
          │                        │                        │                        │
     Dense (32)               Dense (32)              USE / Embedding          Conv1D / Bi-LSTM
          │                        │                        │                        │
          │                        │                        └──────────┬─────────────┘
          │                        │                                   ▼
          │                        │                           Token + Char Hybrid
          │                        │                               Dense (256)
          └────────────────────────┼───────────────────────────────────┘
                                   ▼
                       Tribrid Concatenation Layer
                                   │
                              Dense (5)
                           Softmax Output
```

1. **Pretrained Token Embeddings:** Universal Sentence Encoder (USE) or custom word embeddings for semantic sentence comprehension.
2. **Character-level Encodings:** Character TextVectorization + Conv1D/Bi-LSTM to handle out-of-vocabulary medical nomenclature and typos.
3. **Positional Feature Engineering:** Incorporating normalized `line_number` and `total_lines` as categorical signals (e.g., Objectives appear at the top, Conclusions at the end).

## 📁 Repository Structure
```text
├── 09_SkimLit_nlp_milestone_project_2.ipynb  # Interactive end-to-end milestone pipeline
├── skimlit_model.py                         # Standalone multi-input Keras functional model
├── helper_functions.py                      # Training loop, evaluation, and confusion matrix utilities
├── requirements.txt                         # Dependencies
└── README.md
```

## ⚙️ Setup & Execution

### 1. Installation
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Notebook
Launch Jupyter to explore dataset preprocessing, baseline model benchmarking (Naive Bayes / TF-IDF), and the final multi-input hybrid model:
```bash
jupyter lab 09_SkimLit_nlp_milestone_project_2.ipynb
```

## 📄 Reference
* Franck Dernoncourt, Ji Young Lee. *"PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts"*, 2017. [arXiv:1710.06071](https://arxiv.org/abs/1710.06071)

## 📄 License
MIT License
