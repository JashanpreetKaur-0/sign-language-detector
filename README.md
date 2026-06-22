# 🤟 Real-Time Sign Language Detector

A real-time American Sign Language (ASL) hand sign recognition system that detects 14 ASL letters instantly from a webcam feed using computer vision and machine learning.

Built by **Jashanpreet Kaur** | B.Tech AI & ML | 1st Year

---

## 🎯 What It Does

Show your hand sign to the camera — the AI recognizes it instantly and displays the detected letter on screen in real time.

Supports 14 ASL letters: **A, B, C, D, E, F, G, H, I, L, O, V, W, Y**

---









## 🧠 How It Works

1. **Webcam captures** your hand in real time
2. **MediaPipe** (Google's hand tracking library) detects 21 joint points on your hand
3. Each point has X, Y, Z coordinates → **63 numbers total** representing one hand position
4. These 63 numbers are passed to a **Random Forest classifier**
5. The classifier predicts which letter you're signing
6. Result appears on screen instantly

![Pipeline](https://i.imgur.com/placeholder.png)

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **99.84%** |
| Training samples | **1,400+** |
| Letters supported | **14** |
| Model type | Random Forest (100 trees) |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| MediaPipe | Hand landmark detection |
| OpenCV | Webcam capture and video processing |
| scikit-learn | Random Forest classifier |
| NumPy | Numerical computations |
| Streamlit | Web dashboard |

---

## 📁 Project Structure
sign-language-detector/

│

├── app.py              # Streamlit web dashboard

├── collect_data.py     # Script to collect hand sign training data

├── train_model.py      # Script to train the classifier

├── predict.py          # Real-time prediction script

├── hand_detection.py   # Basic hand detection test

├── check_data.py       # Check collected data counts

├── model.pkl           # Trained Random Forest model

├── requirements.txt    # Python dependencies

└── README.md           # Project documentation

---

## 🚀 Run It Yourself

**1. Clone the repository**
```bash
git clone https://github.com/JashanpreetKaur-0/sign-language-detector.git
cd sign-language-detector
```

**2. Create a virtual environment with Python 3.10**
```bash
py -3.10 -m venv venv310
venv310\Scripts\activate
```

**3. Install dependencies**
```bash
pip install opencv-python mediapipe==0.10.9 numpy==1.24.3 scikit-learn streamlit
```

**4. Run the app**
```bash
streamlit run app.py
```

**5. Tick "Start Camera" and show your hand!**

---

## ✋ Supported Hand Signs

| Letter | Hand Shape |
|--------|-----------|
| A | Fist with thumb on side |
| B | Flat hand, fingers up, thumb tucked |
| C | Curved like holding a cup |
| D | Index finger up, others touch thumb |
| E | All fingers curled down |
| F | Index and thumb touch, 3 fingers up |
| G | Index and thumb pointing sideways |
| H | Index and middle finger sideways |
| I | Only pinky finger up |
| L | Index up, thumb out (gun shape) |
| O | All fingers form a circle |
| V | Index and middle fingers up |
| W | Index, middle, ring fingers up |
| Y | Thumb and pinky out |

---

## 🔮 Future Plans

- [ ] Add remaining ASL alphabet letters
- [ ] Recognize common words (Hello, Yes, No, Thank You)
- [ ] Add text-to-speech output
- [ ] Browser-based version (no installation needed)

---

## 👩‍💻 About Me

I'm a first-year B.Tech AI & ML student passionate about building AI that solves real problems.

![LinkedIn](https://www.linkedin.com/in/jashanpreet-kaur-a6b734381?utm_source=share_via&utm_content=profile&utm_medium=member_android)
![GitHub](https://github.com/JashanpreetKaur-0)

---

## ⭐ If you found this useful, please star the repo!