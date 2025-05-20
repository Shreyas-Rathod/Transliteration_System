# Assignment 3: Sequence-to-Sequence Transliteration

## Course

**DA6401: Deep Learning**
**CS24M046**

## Author

**Shreyas Rathod**

---

## Project Description

This project implements Sequence-to-Sequence models with and without Attention for transliteration tasks using the [Dakshina dataset](https://github.com/google-research-datasets/dakshina).

The goals of this assignment were to:

* Model sequence-to-sequence learning using Recurrent Neural Networks (RNN).
* Compare different recurrent units: vanilla RNN, LSTM, GRU.
* Investigate the effectiveness of Attention mechanisms.
* Visualize and interpret RNN components and activations.

## Project Structure

```plaintext
.
├── Part A - Without Attention
│   ├── Question1.py                 # Model and computations
│   ├── Question2.py                 # Hyperparameter tuning using WandB
│   ├── Question4.py                 # Evaluation on test set
│   ├── gen_conf_matrix.py           # Generate confusion matrix from predictions
│   ├── predictions_vanilla.csv      # Test predictions (vanilla seq2seq)
│   └── predictions_vanilla.txt      # Sample predictions
│
├── PartB - With Attention
│   ├── Question5a.py                # Hyperparameter tuning with attention
│   ├── Question5b.py                # Evaluate attention model on test data
│   ├── Question5d.py                # Generate attention heatmaps
│   ├── Question6.py                 # Connectivity visualization
│   └── predictions_attention.csv    # Test predictions (attention seq2seq)
│
├── LICENSE
└── README.md
```

---

## Running the Code

### Requirements

Ensure you have the following installed:

* Python 3.8+
* PyTorch
* pandas
* matplotlib
* WandB (`wandb`)

```bash
pip install torch pandas matplotlib wandb
```

### Running Scripts

To run the model training and evaluation scripts:

```bash
python Question2.py
python Question4.py
python Question5a.py
python Question5b.py
python Question5d.py
python Question6.py
```

Make sure to update paths to your data in the scripts accordingly.

---

## WandB Experiments & Report Link

All experiment tracking and visualizations were performed using [Weights & Biases](https://wandb.ai/). The visualizations and plots for hyperparameter sweeps and model performance are available in WandB project: `CS24M046_DA6401_A3_t1`.

https://api.wandb.ai/links/cs24m046-indian-institute-of-technology-madras/kb2f5xnu

---

## License

This project is licensed under the [MIT License](LICENSE).

---
