# 🤟 Sign Language Recognition

A real-time sign language recognition system that uses hand pose detection and Graph Neural Networks (GNNs) to classify signs from the American Sign Language (ASL) alphabet (**A–Z**) and digits (**0–9**).


---


## 📸 Demo


<video src="demo/sign_language_alphabet_demo.mov" width="320" height="240" controls></video>
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

```
sign-language-recognition/

├── alphabet_dataset/
│   └── datasets.npy
│
├── model_weights/
│   └── model_weights/model_alphabet_10_epochs.pth
│   └── model_weights/model_numbers_10_epochs.pth
│   └── model_weights/model_numbers_20_epochs.pth
│
├── models/
│   └── models/model_alphabet_train.py
│   └── models/model_alphabet.py
│   └── models/model_number.py
│   └── models/model_numbers_train.py
│
├── number_dataset/
│   └── datasets.npy
│
├── .gitignore
│
└── README.md
│
├── main_alphabet.py
│
├── main_numbers.py
│
└── requirements.txt

```

