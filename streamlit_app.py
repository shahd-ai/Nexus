import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------
# Utility
# ---------------------------
@st.cache_resource
def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Impossible de charger le modèle '{path}': {e}")
        return None

def ensure_columns(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in required_cols:
        if c not in df.columns:
            df[c] = 0
    # keep only required columns and order them
    return df[required_cols]

def one_hot_from_inputs(age:int, bmi:float, children:int, sex:str, smoker:str, region:str) -> pd.DataFrame:
    row = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_female": 1 if sex == "female" else 0,
        "sex_male": 1 if sex == "male" else 0,
        "smoker_no": 1 if smoker == "no" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        "region_northeast": 1 if region == "northeast" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }
    return pd.DataFrame([row])

def csv_download(data: pd.DataFrame, filename: str, label: str):
    return st.download_button(
        label=label,
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )

# ---------------------------
# App
# ---------------------------
st.set_page_config(page_title="Insurance ML App", page_icon="📈", layout="centered")

st.title("Insurance ML • Démo Streamlit")
st.caption("Basé sur votre Notebook. Charge un modèle .pkl et prédit le coût d'assurance.")

# Detect available models from the notebook outputs
default_models = [
    "random_forest_model.pkl",
    "tweedie_cost_model.pkl",
    "charges_tier_classifier.pkl",
    "isolation_forest_fraud_model.pkl"
]
existing_models = [m for m in default_models if os.path.exists(m)]
if not existing_models:
    st.warning("Aucun fichier .pkl trouvé dans le dossier courant. Déposez vos modèles et rechargez la page.")
model_path = st.selectbox("Choisir le modèle", options=existing_models, index=0 if existing_models else None, placeholder="Sélection du modèle")

model = load_model(model_path) if model_path else None

# Feature set inferred from the Notebook
RF_FEATURES = [
    "age","bmi","children",
    "sex_female","sex_male",
    "smoker_no","smoker_yes",
    "region_northeast","region_northwest","region_southeast","region_southwest"
]

with st.expander("Schéma d'entrée attendu", expanded=False):
    st.write(pd.DataFrame({"feature": RF_FEATURES}))

tab1, tab2, tab3 = st.tabs(["🎯 Prédiction unique", "📦 Batch CSV", "🔍 Détection fraude (IsolationForest)"])

with tab1:
    st.subheader("Entrée manuelle")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=0, max_value=120, value=40, step=1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.3, step=0.1)
        children = st.number_input("Children", min_value=0, max_value=10, value=1, step=1)
    with c2:
        sex = st.selectbox("Sex", ["male", "female"], index=0)
        smoker = st.selectbox("Smoker", ["no", "yes"], index=0)
        region = st.selectbox("Region", ["northeast","northwest","southeast","southwest"], index=1)

    if st.button("Predict Insurance Charges", use_container_width=True, type="primary", disabled=model is None):
        X = one_hot_from_inputs(age, bmi, children, sex, smoker, region)
        X = ensure_columns(X, RF_FEATURES)
        if model is not None:
            try:
                y = model.predict(X)[0]
                st.success(f"Charges prédites : **{y:,.2f}**")
            except Exception as e:
                st.error(f"Erreur pendant la prédiction: {e}")

    st.divider()
    st.caption("Télécharger un template CSV")
    template = pd.DataFrame([{
        'age': 40, 'bmi': 25.3, 'children': 1,
        'sex_female': 0, 'sex_male': 1,
        'smoker_no': 1, 'smoker_yes': 0,
        'region_northeast': 0, 'region_northwest': 1, 'region_southeast': 0, 'region_southwest': 0
    }])
    csv_download(template, "template_inputs.csv", "📥 Télécharger le template")

with tab2:
    st.subheader("Prédiction par lot (CSV)")
    file = st.file_uploader("Déposer un CSV avec les colonnes requises", type=["csv"])
    if file is not None and model is not None:
        try:
            df = pd.read_csv(file)
            X = ensure_columns(df, RF_FEATURES)
            preds = model.predict(X)
            out = df.copy()
            out["predicted_charges"] = preds
            st.dataframe(out, use_container_width=True)
            csv_download(out, "batch_predictions.csv", "📤 Télécharger les prédictions")
        except Exception as e:
            st.error(f"Erreur de traitement du CSV: {e}")
    elif model is None:
        st.info("Sélectionnez un modèle pour activer cette section.")

with tab3:
    st.subheader("Score d'anomalie (fraude)")
    if "isolation_forest_fraud_model.pkl" in existing_models:
        iso = load_model("isolation_forest_fraud_model.pkl")
        st.caption("Utilise age, bmi, children, charges. Renvoie 1 = anomalie, 0 = normal.")
        c1, c2 = st.columns(2)
        with c1:
            age_f = st.number_input("Age (fraude)", 0, 120, 40, 1)
            bmi_f = st.number_input("BMI (fraude)", 0.0, 80.0, 25.0, 0.1)
        with c2:
            children_f = st.number_input("Children (fraude)", 0, 10, 1, 1)
            charges_f = st.number_input("Charges observées", 0.0, 100000.0, 5000.0, 10.0)

        if st.button("Détecter", use_container_width=True, disabled=iso is None):
            try:
                Xf = pd.DataFrame([{"age": age_f, "bmi": bmi_f, "children": children_f, "charges": charges_f}])
                # Some notebooks map: 1->0 normal, -1->1 anomaly. We produce final label accordingly.
                raw = iso.predict(Xf)[0]  # 1 normal, -1 anomaly in sklearn
                label = 1 if raw == -1 else 0
                st.info(f"Label fraude: **{label}**  (1 = anomalie, 0 = normal)")
            except Exception as e:
                st.error(f"Erreur détection: {e}")
    else:
        st.info("Modèle d'Isolation Forest introuvable. Placez 'isolation_forest_fraud_model.pkl' dans le dossier pour activer.")
