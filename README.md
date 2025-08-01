# Encoding Time: A Comparison of Temporal Representation in RNNs, LSTMs, and Transformers

This repository contains the experimental codebase for the paper:

> **Encoding Time: A Comparison of Temporal Representation in RNNs, LSTMs and Transformers**  
> *Author: Youssef Soula – July 2025*

The project investigates how different neural architectures represent and process **temporal dependencies** in sequential data, through a combination of **theoretical analysis** and **controlled synthetic experiments**.

---

## 🧠 Overview

Three well-known architectures are compared:

- **Recurrent Neural Networks (RNNs)**
- **Long Short-Term Memory networks (LSTMs)**
- **Transformer Encoders**

The focus is on understanding how each model handles **time-dependent tasks** — memory, delay, and pattern recognition — when real-world noise is removed.

---

## 🧪 Diagnostic Tasks

| Task                | Goal                                     | Skill Tested                    |
|---------------------|------------------------------------------|----------------------------------|
| **Copying Task**     | Memorize & reproduce a delayed sequence | Long-term memory                 |
| **Matching Task**    | Recall a token from earlier              | Memory + attention over delay    |
| **Pattern Completion** | Predict next token from recent history | Local temporal pattern recognition |

All tasks are synthetic, allowing for isolated evaluation of temporal reasoning.

---

## 📁 Project Structure

experiment/
├── datasets/ # Synthetic data generators
│ ├── copying.py
│ ├── matching.py
│ └── pattern_completion.py
├── models/ # Model definitions
│ ├── rnn.py
│ ├── lstm.py
│ └── transformer.py
├── results/ # Output plots (loss curves)
│ ├── copying_loss.png
│ ├── matching_loss.png
│ └── pattern_loss.png
├── train.py # Training loop
├── evaluate.py # Evaluation + plotting
├── main.py # Main runner for training + evaluation
├── README.md # Project documentation
└── .gitignore

## 🧰 Requirements

Install dependencies with:

```bash
pip install torch matplotlib numpy

