# Real-Time Language Translation Path Learning

This project focuses on developing and comparing machine learning algorithms for optimizing real-time language translation paths using bilingual corpora. We explore how different approaches such as **Q-Learning**, **SARSA**, **Artificial Neural Networks (ANN)**, and **Linear Regression** can enhance translation accuracy and efficiency between multiple languages.

## 🚀 Objective

To model the process of selecting optimal translation paths across languages in real-time, inspired by reinforcement learning and supervised learning techniques.

## 📚 Datasets

We utilize two widely used multilingual corpora:

- **TED Talks Corpus**: A large set of parallel transcriptions of TED Talks in multiple languages.
- **Europarl Corpus**: Proceedings from the European Parliament, translated into 21 European languages.

These datasets allow the construction of a multilingual translation graph, where nodes are languages and edges represent available translations.

## 🧠 Methods Implemented

- **Q-Learning**: Reinforcement learning algorithm for discovering optimal translation paths without prior knowledge of environment dynamics.
- **SARSA (State–Action–Reward–State–Action)**: On-policy reinforcement learning variant used for path evaluation.
- **Artificial Neural Networks (ANNs)**: Used to predict translation accuracy or likelihood based on contextual features.
- **Linear Regression**: Used as a baseline model to score translation paths based on learned weights.

## 🔄 Translation Path Concept

Languages are represented as nodes in a graph, and direct translations between them as edges. The goal is to find the **optimal path** (sequence of languages) to translate a sentence from a source to a target language that maximizes translation quality or minimizes loss.

Example:
```text
English ➝ German ➝ French
