import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import Counter

# page config
st.set_page_config(
    page_title="Sign Language Detector",
    page_icon="🤟",
    layout="wide"        # uses full width of screen
)

# custom CSS for colors and styling
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    .title {
        font-size: 48px;
        font-weight: bold;
        color: #00d4ff;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 18px;
        color: #aaaaaa;
        text-align: center;
        margin-bottom: 30px;
    }
    .detected-box {
        background-color: #1e2d40;
        border: 2px solid #00d4ff;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin-top: 20px;
    }
    .detected-letter {
        font-size: 120px;
        font-weight: bold;
        color: #00ff88;
        line-height: 1;
    }
    .detected-label {
        font-size: 20px;
        color: #aaaaaa;
        margin-top: 10px;
    }
    .letter-grid {
        background-color: #1a1a2e;
        border-radius: 10px;
        padding: 15px;
        margin-top: 20px;
    }
    .footer {
        text-align: center;
        color: #555555;
        font-size: 13px;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# title section
st.markdown('<div class="title">🤟 Sign Language Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Show your hand sign to the camera — AI will recognize it in real time</div>', unsafe_allow_html=True)

st.divider()

# load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# two column layout — camera on left, result on right
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📷 Live Camera Feed")
    run = st.checkbox("▶ Start Camera", value=False)
    FRAME_WINDOW = st.image([])

with col2:
    st.markdown("### 🔍 Detected Sign")
    result_box = st.empty()

    st.markdown("### 🔤 Supported Letters")
    # show all 14 supported letters in a nice grid
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "L", "O", "V", "W", "Y"]
    cols = st.columns(7)
    for i, letter in enumerate(letters):
        with cols[i % 7]:
            st.markdown(f"""
                <div style='background:#1e2d40; border:1px solid #00d4ff;
                border-radius:8px; text-align:center; padding:8px;
                color:#00d4ff; font-weight:bold; font-size:18px;
                margin-bottom:5px;'>
                {letter}
                </div>
            """, unsafe_allow_html=True)

    st.markdown("### 📖 How to use")
    st.markdown("""
    1. Click **Start Camera** above
    2. Hold your hand clearly in frame
    3. Make one of the 14 signs
    4. Hold still for 1-2 seconds
    5. See the result appear here!
    """)

# camera loop
recent_predictions = []
cap = cv2.VideoCapture(0)

while run:
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    stable_letter = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            row = []
            for point in hand_landmarks.landmark:
                row.append(point.x)
                row.append(point.y)
                row.append(point.z)

            row = np.array(row).reshape(1, -1)
            prediction = model.predict(row)[0]

            recent_predictions.append(prediction)
            if len(recent_predictions) > 15:
                recent_predictions.pop(0)

            most_common = Counter(recent_predictions).most_common(1)[0]
            if most_common[1] >= 10:
                stable_letter = most_common[0]

    else:
        recent_predictions = []

    # show camera feed
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # show result in styled box
    if stable_letter:
        result_box.markdown(f"""
            <div class="detected-box">
                <div class="detected-letter">{stable_letter}</div>
                <div class="detected-label">Sign detected!</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        result_box.markdown(f"""
            <div class="detected-box">
                <div style="font-size:40px;">👋</div>
                <div class="detected-label">Show your hand...</div>
            </div>
        """, unsafe_allow_html=True)

else:
    cap.release()
    result_box.markdown(f"""
        <div class="detected-box">
            <div style="font-size:40px;">⏸️</div>
            <div class="detected-label">Camera stopped</div>
        </div>
    """, unsafe_allow_html=True)

# footer
st.markdown('<div class="footer">Built with MediaPipe + Random Forest | By Jashanpreet Kaur</div>',
            unsafe_allow_html=True)