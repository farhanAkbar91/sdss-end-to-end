import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# Configuration
st.set_page_config(page_title="SDSS DR17 Classifier", page_icon="🌌", layout="wide")

# Custom CSS for dark theme adjustments
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #4f86c6;
    }
    .glossary-box {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4f86c6;
        margin-top: 20px;
    }
    .metric-card {
        background-color: #1e1e2e;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Main Title
st.markdown("<h1 class='main-header'>🌌 SDSS Stellar Classification Dashboard</h1>", unsafe_allow_html=True)
st.markdown("Classify celestial objects (Stars, Galaxies, Quasars) using Machine Learning based on SDSS data.")

# Sidebar for Input
st.sidebar.header("Telescope Observations (Features)")

with st.sidebar.expander("Coordinates & Observation Details", expanded=True):
    alpha = st.number_input("Right Ascension (alpha)", value=180.0, format="%.4f")
    delta = st.number_input("Declination (delta)", value=0.0, format="%.4f")
    redshift = st.number_input("Redshift", value=0.0, format="%.4f")
    cam_col = st.number_input("Camera Column (cam_col)", min_value=1, max_value=6, value=3)
    plate = st.number_input("Plate ID", value=4000, step=1)
    mjd = st.number_input("Modified Julian Date (MJD)", value=55000, step=1)

with st.sidebar.expander("Photometric Filters (ugriz)", expanded=True):
    u = st.number_input("u (Ultraviolet)", value=23.0, format="%.2f")
    g = st.number_input("g (Green)", value=21.0, format="%.2f")
    r = st.number_input("r (Red)", value=20.0, format="%.2f")
    i = st.number_input("i (Near Infrared)", value=19.0, format="%.2f")
    z = st.number_input("z (Infrared)", value=18.0, format="%.2f")

# Build the feature payload
features = {
    "alpha": alpha, "delta": delta, "u": u, "g": g, "r": r, "i": i, "z": z,
    "cam_col": cam_col, "redshift": redshift, "plate": plate, "MJD": mjd
}

# Glossary Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Astronomy Glossary")
with st.sidebar.expander("View Definitions"):
    st.markdown("""
    - **Redshift:** Measure of how much the light from an object is shifted toward the red end of the spectrum, indicating its velocity away from Earth (expansion of the universe). High redshift often implies distant objects like Quasars.
    - **ugriz Filters:** The five filters used by SDSS to measure the brightness of objects in different wavelengths:
        - **u:** Ultraviolet
        - **g:** Green
        - **r:** Red
        - **i:** Near Infrared
        - **z:** Infrared
    - **MJD (Modified Julian Date):** The date the observation was made, counted in days from a specific starting point.
    - **Plate ID:** ID of the aluminum plate used to hold optical fibers for spectroscopy.
    - **Alpha & Delta:** Right Ascension and Declination. The celestial equivalent of longitude and latitude.
    """)

# Main Content Area
tab1, tab2 = st.tabs(["🎯 Single Prediction", "📊 Model Comparison"])

FLASK_URL = "http://localhost:5000"

with tab1:
    st.subheader("Single Model Prediction")
    selected_model = st.selectbox("Select ML Model", ["Random Forest", "Decision Tree", "Logistic Regression"])
    
    if st.button("Predict Object", type="primary"):
        with st.spinner("Analyzing spectral data..."):
            try:
                res = requests.post(f"{FLASK_URL}/predict", json={"features": features, "model": selected_model})
                
                if res.status_code == 200:
                    data = res.json()
                    prediction = data['prediction']
                    probabilities = data.get('probabilities', {})
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f"<div class='metric-card'><h3>Prediction</h3><h1 style='color: #4f86c6;'>{prediction}</h1><p>Model: {selected_model}</p></div>", unsafe_allow_html=True)
                        
                        if prediction == "STAR":
                            st.image("https://images.unsplash.com/photo-1519681393784-d120267933ba", caption="Star")
                        elif prediction == "GALAXY":
                            st.image("https://images.unsplash.com/photo-1462331940025-496dfbfc7564", caption="Galaxy")
                        elif prediction == "QSO":
                            st.image("https://images.unsplash.com/photo-1465101162946-4377e57745c3", caption="Quasar (QSO)")
                            
                    with col2:
                        if probabilities:
                            st.write("### Prediction Probabilities")
                            prob_df = pd.DataFrame(list(probabilities.items()), columns=['Class', 'Probability'])
                            fig = px.bar(prob_df, x='Probability', y='Class', orientation='h', 
                                         color='Class', text='Probability',
                                         color_discrete_map={'GALAXY': '#4f86c6', 'QSO': '#e06c75', 'STAR': '#e5c07b'})
                            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Error from API: {res.text}")
            except Exception as e:
                st.error(f"Connection error: {e}. Is the Flask backend running?")

with tab2:
    st.subheader("Compare All Models")
    if st.button("Run Comparison", type="primary", key="compare_btn"):
        with st.spinner("Evaluating all models..."):
            try:
                res = requests.post(f"{FLASK_URL}/compare", json={"features": features})
                
                if res.status_code == 200:
                    data = res.json()
                    results = data['results']
                    
                    st.write("### Prediction Results")
                    cols = st.columns(3)
                    for i, (model_name, info) in enumerate(results.items()):
                        with cols[i]:
                            pred = info['prediction']
                            acc = info['metrics'].get('accuracy', 0) * 100
                            f1 = info['metrics'].get('f1_score', 0) * 100
                            
                            st.markdown(f"""
                            <div class='metric-card'>
                                <h4 style='color: #4f86c6;'>{model_name}</h4>
                                <p>Prediction: <strong>{pred}</strong></p>
                                <hr style='border-color: #444;'>
                                <p>Accuracy: {acc:.2f}%</p>
                                <p>F1-Score: {f1:.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    # Show comparison chart of metrics
                    st.write("### Performance Metrics Comparison")
                    metrics_data = []
                    for model_name, info in results.items():
                        metrics_data.append({
                            'Model': model_name,
                            'Accuracy': info['metrics'].get('accuracy', 0),
                            'F1 Score': info['metrics'].get('f1_score', 0)
                        })
                    m_df = pd.DataFrame(metrics_data)
                    m_df_melted = pd.melt(m_df, id_vars=['Model'], value_vars=['Accuracy', 'F1 Score'], var_name='Metric', value_name='Score')
                    
                    fig = px.bar(m_df_melted, x='Model', y='Score', color='Metric', barmode='group',
                                 color_discrete_sequence=['#4f86c6', '#e06c75'])
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', yaxis_range=[0.8, 1.0])
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error(f"Error from API: {res.text}")
            except Exception as e:
                st.error(f"Connection error: {e}. Is the Flask backend running?")