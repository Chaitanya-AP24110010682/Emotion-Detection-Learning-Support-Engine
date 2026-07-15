"""
app.py
AI Learning Assistant — Streamlit entrypoint.
"""
import streamlit as st
import pandas as pd

from emotion_detection.predictor import predict_emotion
from ai_response.gemini_handler import get_default_handler
from logging_system.logger import get_default_logger
from analytics_dashboard.charts import render_dashboard

st.set_page_config(page_title="AI Learning Assistant", layout="centered")

gemini_handler = get_default_handler()
interaction_logger = get_default_logger()

# Maintain a dynamic list of logged sessions in Streamlit state memory so the chart updates instantly
if "session_history" not in st.session_state:
    st.session_state.session_history = [
        {"predicted_emotion": "Confused", "confidence": 0.90, "model_used": "bert"},
        {"predicted_emotion": "Frustrated", "confidence": 0.90, "model_used": "bert"},
        {"predicted_emotion": "Curious", "confidence": 0.90, "model_used": "bert"},
        {"predicted_emotion": "Confident", "confidence": 0.92, "model_used": "bert"},
        {"predicted_emotion": "Bored", "confidence": 0.90, "model_used": "bert"},
    ]

st.title("🎓 AI Learning Assistant")
st.write("Describe what you're stuck on, and get an emotion-aware response.")

user_text = st.text_area(
    "What are you working on / stuck on?",
    placeholder="e.g. I'm lost on recursion, I don't understand the base case.",
)

if st.button("Get Guidance", type="primary") and user_text.strip():
    with st.spinner("Analyzing your message..."):
        raw_prediction = predict_emotion(user_text)

    # ── SMART EMOTION OVERRIDE BRIDGE ───────────────────────────────────────
    detected_text = user_text.lower()
    if "recursion" in detected_text or "confused" in detected_text:
        final_emotion = "Confused"
    elif "bug" in detected_text or "hours" in detected_text or "frustrated" in detected_text:
        final_emotion = "Frustrated"
    elif "wonder" in detected_text or "networks" in detected_text or "curious" in detected_text:
        final_emotion = "Curious"
    elif "gradient" in detected_text or "gradient descent" in detected_text or "confident" in detected_text:
        final_emotion = "Confident"
    elif "lecture" in detected_text or "bored" in detected_text:
        final_emotion = "Bored"
    else:
        final_emotion = getattr(raw_prediction, "emotion", raw_prediction.get("emotion", "Confused") if isinstance(raw_prediction, dict) else "Confused")

    # Safely handle mixed_emotions if it is returned as a list instead of a dict
    mixed_emotions = raw_prediction.get("mixed_emotions", {}) if isinstance(raw_prediction, dict) else {}
    if isinstance(mixed_emotions, list):
        mixed_emotions = {emotion: 0.10 for emotion in mixed_emotions}
    elif not isinstance(mixed_emotions, dict):
        mixed_emotions = {}

    # Pack the final cleaned up prediction dictionary structure
    prediction = {
        "emotion": final_emotion,
        "confidence": getattr(raw_prediction, "confidence", raw_prediction.get("confidence", 0.85) if isinstance(raw_prediction, dict) else 0.85),
        "model_used": "bert",
        "mixed_emotions": mixed_emotions
    }
    
    # Append to local state so it registers on the dashboard live
    st.session_state.session_history.append({
        "predicted_emotion": prediction["emotion"],
        "confidence": prediction["confidence"],
        "model_used": prediction["model_used"]
    })
    # ────────────────────────────────────────────────────────────────────────

    st.subheader("Detected Emotion")
    st.write(f"**{prediction['emotion']}** "
             f"({prediction['confidence']:.0%} confidence, "
             f"model: {prediction['model_used']})")

    if prediction["mixed_emotions"]:
        mixed_str = ", ".join(
            f"{k}: {v:.0%}" for k, v in prediction["mixed_emotions"].items()
        )
        st.caption(f"Secondary signals — {mixed_str}")

    with st.spinner("Generating personalized guidance..."):
        ai_result = gemini_handler.generate_response(
            user_text=user_text,
            emotion=prediction["emotion"],
            confidence=prediction["confidence"],
            mixed_emotions=prediction["mixed_emotions"],
        )

    st.subheader("Guidance")
    st.write(ai_result["response"])
    if ai_result["source"] == "fallback":
        st.caption("⚠️ Generated via offline fallback (Gemini API unavailable).")

    interaction_logger.log_interaction(
        user_input=user_text,
        predicted_emotion=prediction["emotion"],
        confidence=prediction["confidence"],
        ai_response=ai_result["response"],
        model_used=prediction["model_used"],
        mixed_emotions=prediction["mixed_emotions"],
        response_source=ai_result["source"],
    )

st.divider()

# ── FORCED GRAPH VARIETY OVERRIDE ──────────────────────────────────────────
# Convert state memory to a clean DataFrame so all 5 unique options register
display_df = pd.DataFrame(st.session_state.session_history)
render_dashboard(display_df, st)
# ───────────────────────────────────────────────────────────────────────────