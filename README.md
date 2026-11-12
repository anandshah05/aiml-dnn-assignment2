# Deep Neural Network (DNN) Programming Assignment 2

## RNN, LSTM, GRU and Transformer from Scratch

-   **Student Name:** Anand Shah
-   **Student ID:** 2026AJ05251
-   **Chosen Variants & Hyperparameters:**
    -   **Problem 1**:
        -   Hidden Size:** h = 64 (applied uniformly across RNN,
        -   **Seeds for 3-seed runs:** 42, 43, 44
        LSTM, and GRU)
    -   **Problem 2**:
        - Model Width:** d = 128 (divisible by num_heads: d /
        4 = 32)
        -   **Feed-Forward Dimension:** d_ff = 4*d = 512
        -   **Seeds for 3-seed runs:** 42, 43, 44

### Datasets and environment

-   For problem 1,
    [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/humanactivityrecognitionusingsmartphones)
    dataset is used.
-   For problem 2,
    [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
    input text is used.

### Setup and Installation

```bash
pip install -r requirements.txt
```

If not aliased, use pip3 directly 

```bash
pip3 install -r requirements.txt
```


### Directory structure

```
<StudentID>_DNN_Assignment/
├── README.md
├── requirements.txt
├── report.pdf
├── results.csv
├── problem1/
│   ├── problem1.ipynb
│   └── figures/
└── problem2/
    ├── problem2.ipynb
    ├── generated_samples.txt
    └── figures/
```

### How to Run Each Notebook

1.  **Jupyter / Colab Environment:**
    -   Open Jupyter Lab or Google Colab
        (`Runtime -> Change runtime type -> T4 GPU` or CPU).
    -   In Colab, verify GPU availability: `torch.cuda.is_available()`.
2.  **Problem 1 Execution:**
    -   Open `problem1/problem1.ipynb`.
    -   Click `Kernel -> Restart & Run All` (or `Runtime -> Run all`).
    -   The notebook automatically downloads the UCI HAR dataset and generates
        the synthetic adding problem, executes unit test assertions, trains
        models across seeds 42, 43, 44, and saves all evaluation plots.
3.  **Problem 2 Execution:**
    -   Open `problem2/problem2.ipynb`.
    -   Click `Kernel -> Restart & Run All`.
    -   The notebook downloads Tiny Shakespeare, executes all unit tests for
        scaled dot-product attention (including 3-token hand check
        verification), trains the Pre-LN Transformer encoder classifier and
        causal language model, performs ablations, and generates text samples.

### Implementation Notes

-   **All Unit Tests Pass Deterministically:** Unit tests for MyRNNCell,
    MyLSTMCell, MyGRUCell, scaled dot-product attention, causal masking,
    row-stochasticity, and hand check 3-token example match PyTorch library
    within $10^{-5}$.
-   **No Failures / Broken Code:** All parts run top to bottom without error.
-   **Followed Assignment Requirements:**
    -   Problem 1 Task 2 uses strictly matrix operations (`@`, `+`,
        `torch.sigmoid`, `torch.tanh`) with no `nn.RNN`, `nn.LSTM`, or `nn.GRU`
        cell classes inside.
    -   Problem 2 uses strictly custom `scaled_dot_product_attention`,
        `MultiHeadAttention`, and `TransformerEncoderBlock` tensor operations.
        No `nn.MultiheadAttention` or `nn.Transformer*` are used.

