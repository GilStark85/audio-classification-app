# 🎧 Audio Classification App

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Classifies uploaded audio as:
- 🎤 Dialogue (Speech)
- 🎶 Music
- 💥 Sound Effects

Built with machine learning using Python and Streamlit.

---

## 📸 Demo Screenshot

> Add your own screenshot here!
![Screenshot](screenshots/app_preview.png)

---

## 🚀 Features

- Real-time audio classification via browser
- MFCC feature extraction using Librosa
- Random Forest classifier (scikit-learn)
- Live prediction with class probabilities

---

## 🌐 [Try It Live](https://audio-classification-app-hxmtmr7u4n4qpuaf974x2l.streamlit.app/)

---

## 📁 Project Structure

```
├── app.py
├── train_audio_classifier.py
├── audio_model.pkl
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── tests/
    └── test_app.py
```

---

## 🧠 Tech Stack

- [Python 3.10](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [Librosa](https://librosa.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/urbansound8k.html)

---

## 🛠️ Run Locally

```bash
git clone https://github.com/GilStark85/audio-classification-app.git
cd audio-classification-app
pip install -r requirements.txt
streamlit run app.py
```

---

## 📬 Contact

Got questions or want to collaborate? [Open an issue](https://github.com/GilStark85/audio-classification-app/issues)

---
 model             |
| `requirements.txt`        | Python dependencies                        |
| `tests/test_app.py`       | Unit test for model and feature extraction |
| `demo/`                   | Contains screenshots and GIFs              |

---

## 🧠 Tech Stack

- [Python 3.10](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [Librosa](https://librosa.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/urbansound8k.html)

---

## 🔮 Roadmap

- [ ] Add real-time microphone input
- [ ] Display waveform and spectrogram visualizations
- [ ] Support MP3 to WAV conversion behind-the-scenes
- [ ] Upgrade model to CNN or transformer-based classifier

---

## 🌟 Featured Project

> This app is part of my AI portfolio focused on entertainment and post-production tech.

[![View the Repo](https://img.shields.io/badge/GitHub-View%20Code-black?logo=github)](https://github.com/GilStark85/audio-classification-app)
[![Try Live](https://img.shields.io/badge/Streamlit-Try%20App-brightgreen?logo=streamlit)](https://audio-classification-app-hxmtmr7u4n4qpuaf974x2l.streamlit.app/)

---

## 🛠️ Run Locally

Clone the repository and install dependencies:

```bash
git clone https://github.com/GilStark85/audio-classification-app.git
cd audio-classification-app
pip install -r requirements.txt
streamlit run app.py
```

---

## 📬 Contact

If you have any questions or feedback, feel free to reach out via GitHub issues.

---
