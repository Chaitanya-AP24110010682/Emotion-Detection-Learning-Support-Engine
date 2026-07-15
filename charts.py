# src/analytics_dashboard/charts.py

import plotly.express as px
import pandas as pd

def render_dashboard(stats, st):
    """
    Generates and renders analytical chart panels within the lower application layer.
    
    Parameters:
    - stats: The DataFrame or dictionary from interaction_logger.summary_stats()
    - st: The Streamlit module context passed from app.py to display the charts.
    """
    st.header("📊 Learning Analytics Dashboard")
    
    # 1. Handle case where the log file is completely empty
    if stats is None or (isinstance(stats, pd.DataFrame) and stats.empty):
        st.info("Awaiting interactive log streaming data points. Run a few prompts to generate graphs!")
        return

    # 2. Safety check: Convert to DataFrame if passed as a dictionary
    df = pd.DataFrame(stats) if not isinstance(stats, pd.DataFrame) else stats

    # Create a 2-column layout for side-by-side charts
    col1, col2 = st.columns(2)
    
    # --- CHART 1: Emotion Distribution State ---
    with col1:
        if "predicted_emotion" in df.columns:
            counts = df["predicted_emotion"].value_counts().reset_index()
            counts.columns = ["Emotion State", "Log Volume"]
            
            fig1 = px.bar(
                counts, 
                x="Emotion State", 
                y="Log Volume", 
                color="Emotion State", 
                title="Distribution of Learning States", 
                text_auto=True
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning("Column 'predicted_emotion' not found in logs.")
            
    # --- CHART 2: Confidence Spread (BiLSTM vs BERT) ---
    with col2:
        if "confidence" in df.columns and "model_used" in df.columns:
            fig2 = px.box(
                df, 
                x="model_used", 
                y="confidence", 
                color="model_used", 
                title="Prediction Accuracy Spread (BiLSTM vs BERT)"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Columns 'confidence' or 'model_used' missing from logs.")