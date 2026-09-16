import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb

st.set_page_config(page_title="Prédiction état véhicule", page_icon="🚗")

# ----------------------------
# Chargement des objets sauvegardés
# ----------------------------
@st.cache_resource
def load_objects():
    encoders = jb.load("encoders.joblib")   # [Marque, Transmission, Quartier, Etat]
    uniques = jb.load("uniques.joblib")     # [Marque, Transmission, Quartier, Etat]
    scaler = jb.load("scaler.joblib")
    model = jb.load("gb_model.joblib")
    return encoders, uniques, scaler, model

encoders, uniques, scaler, model = load_objects()

clasnames = uniques[3]  # noms des classes : Occasion / Venant

# ----------------------------
# Fonction de prédiction simple
# ----------------------------
def pred_func(marque, annee, transmission, prix, quartier):
    marque_enc = encoders[0].transform([marque])[0]
    transmission_enc = encoders[1].transform([transmission])[0]
    quartier_enc = encoders[2].transform([quartier])[0]

    # Ordre d'entrainement : Marque, Année, Transmission, Quartier, Prix
    x_new = np.array([[marque_enc, annee, transmission_enc, quartier_enc, prix]])
    x_new = scaler.transform(x_new)

    y_pred = model.predict(x_new)
    return clasnames[y_pred[0]]

# ----------------------------
# Fonction de prédiction multiple (CSV)
# ----------------------------
def pred_func_csv(df):
    predictions = []
    for _, row in df.iterrows():
        pred = pred_func(row["Marque"], row["Année"], row["Transmission"], row["Prix"], row["Quartier"])
        predictions.append(pred)
    df["Etat"] = predictions
    return df

# ----------------------------
# Interface
# ----------------------------
st.title("🚗 Prédire l'état d'un véhicule")
st.write(
    "Ce modèle permet de prédire si un véhicule est **Occasion** ou **Venant**, "
    "en partant de la marque, de l'année, de la transmission, du prix et du quartier."
)

tab1, tab2 = st.tabs(["Prédiction simple", "Prédiction multiple (CSV)"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        marque = st.selectbox("Marque", options=list(uniques[0]))
        annee = st.number_input("Année", min_value=1990, max_value=2026, value=2015, step=1)
        transmission = st.selectbox("Transmission", options=list(uniques[1]))
    with col2:
        prix = st.number_input("Prix", min_value=0, value=1000000, step=10000)
        quartier = st.selectbox("Quartier", options=list(uniques[2]))

    if st.button("Prédire", type="primary"):
        try:
            resultat = pred_func(marque, annee, transmission, prix, quartier)
            st.success(f"État prédit : **{resultat}**")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

with tab2:
    st.write("Le fichier CSV doit contenir les colonnes : `Marque`, `Année`, `Transmission`, `Prix`, `Quartier`.")
    fichier = st.file_uploader("Importer un fichier CSV", type=["csv"])
    if fichier is not None:
        try:
            df_input = pd.read_csv(fichier)
            df_result = pred_func_csv(df_input.copy())
            st.dataframe(df_result)
            csv_out = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger les prédictions",
                data=csv_out,
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")
