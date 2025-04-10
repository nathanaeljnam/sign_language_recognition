# 🤟 Sign Language Recognition

A real-time sign language recognition system that uses hand pose detection and Graph Neural Networks (GNNs) to classify signs from the American Sign Language (ASL) alphabet (**A–Z**) and digits (**0–9**).

<!-- ![Sign Language Example](https://user-images.githubusercontent.com/your-placeholder/example.gif) Optional: add demo gif/image -->

---

## 📌 Features

- ✋ Real-time hand tracking with **MediaPipe** and **OpenCV**
- 🧠 Deep learning with **Graph Neural Networks**
- 🔤 Classifies ASL alphabet signs (A-Z)
- 🔢 Supports digits 0–9
- 🎥 Live webcam detection

---

## 🛠️ Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

---

## 🚀 Getting Started

Clone the repo:

```bash
git clone https://github.com/yourusername/sign-language-recognition.git
cd sign-language-recognition
```

To recognize alphabet signs:

```bash
python main_alphabet.py
```
To recognize number signs:


```bash
python main_numbers.py
```

Make sure your webcam is enabled.
Press Q to quit the app.

---

## 📂 Project Structure

sign-language-recognition/
│
├── main_alphabet.py               # Real-time alphabet recognition script
├── main_numbers.py                # Real-time numbers recognition script
│
├── models/
│   └── model_alphabet.py          # GNN model definition
│
├── model_weights/
│   └── model_alphabet_10_epochs.pth  # Trained model weights
│
├── utils/                         # (Optional) Helper scripts/functions
│
└── README.md                      # This file!

---

## 📸 Demo


