import streamlit as st

st.set_page_config(page_title="Multibagger AI", layout="wide")
st.title("🔍 Loading Multibagger AI App...")

try:
    import pandas as pd
    import pickle
    from src.feature_engineering import generate_features
    from src.model import predict_multibaggers

    st.success("✅ Modules loaded")

    data = pd.read_csv("data/penny_stocks_sample.csv")
    st.success("✅ Data loaded")

    model = pickle.load(open("models/model.pkl", "rb"))
    st.success("✅ Model loaded")

    features = generate_features(data)
    predictions = predict_multibaggers(model, features)

    results = data.copy()
    results["Prediction"] = predictions["label"]
    results["Confidence"] = predictions["confidence"]
    results["Target Price (1Y)"] = predictions["target_price"]

    st.dataframe(results)

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
