# Importation des modules
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Configuration des options Streamlit
st.set_page_config(layout="wide")

# Fonction pour recueillir les caractéristiques utilisateur via la barre latérale
def caracteristiques_utilisateur():
    st.sidebar.header("Caractéristiques du Patient")

    # Sélection des caractéristiques par l'utilisateur
    sex = st.sidebar.selectbox("Sexe", options=["Femme", "Homme"])
    age = st.sidebar.slider("Âge du patient", min_value=0, max_value=120, value=30)
    patient_type = st.sidebar.selectbox("Type de patient", options=["Non hospitalisé", "Hospitalisé"])
    pneumonia = st.sidebar.selectbox("Pneumonie", options=["Non", "Oui"])
    diabetes = st.sidebar.selectbox("Diabète", options=["Non", "Oui"])
    copd = st.sidebar.selectbox("Bronchopneumopathie chronique obstructive (COPD)", options=["Non", "Oui"])
    asthma = st.sidebar.selectbox("Asthme", options=["Non", "Oui"])
    inmsupr = st.sidebar.selectbox("Immunodéprimé", options=["Non", "Oui"])
    hypertension = st.sidebar.selectbox("Hypertension", options=["Non", "Oui"])
    cardiovascular = st.sidebar.selectbox("Maladies cardiovasculaires", options=["Non", "Oui"])
    renal_chronic = st.sidebar.selectbox("Maladies rénales chroniques", options=["Non", "Oui"])
    other_disease = st.sidebar.selectbox("Autres maladies", options=["Non", "Oui"])
    obesity = st.sidebar.selectbox("Obésité", options=["Non", "Oui"])
    tobacco = st.sidebar.selectbox("Tabagisme", options=["Non", "Oui"])
    intubed = st.sidebar.selectbox("Patient intubé", options=["Non", "Oui"])
    icu = st.sidebar.selectbox("Admission en soins intensifs", options=["Non", "Oui"])

    # Création d'un dictionnaire pour stocker les données
    donnees = {
        'sex': sex,
        'age': age,
        'patient_type': patient_type,
        'pneumonia': pneumonia,
        'diabetes': diabetes,
        'copd': copd,
        'asthma': asthma,
        'inmsupr': inmsupr,
        'hypertension': hypertension,
        'cardiovascular': cardiovascular,
        'renal_chronic': renal_chronic,
        'other_disease': other_disease,
        'obesity': obesity,
        'tobacco': tobacco,
        'intubed': intubed,
        'icu': icu
    }

    # Conversion en DataFrame
    caracteristiques = pd.DataFrame(donnees, index=[0])
    return caracteristiques

# Introduction de l'application web
st.write("""
# Application Web : Prédiction du Risque COVID-19
## À propos de cette application :
Cette application utilise un modèle de classification entraîné pour prédire si un patient est à **haut risque** ou **faible risque** de complications liées au COVID-19.
""")

# Collecte des caractéristiques utilisateur
df_entree = caracteristiques_utilisateur()

# Encodage des variables catégoriques pour le modèle
def encode_input(value, positive="Oui", negative="Non"):
    return 1 if value == positive else 2  # "Oui" = 1, "Non" = 2

df_entree_encoded = df_entree.copy()
df_entree_encoded['sex'] = df_entree_encoded['sex'].apply(lambda x: 1 if x == "Femme" else 2)
df_entree_encoded['patient_type'] = df_entree_encoded['patient_type'].apply(lambda x: 1 if x == "Hospitalisé" else 2)
for col in [
    'pneumonia', 'diabetes', 'copd', 'asthma', 'inmsupr', 'hypertension',
    'cardiovascular', 'renal_chronic', 'other_disease', 'obesity', 'tobacco', 
    'intubed', 'icu'
]:
    df_entree_encoded[col] = df_entree_encoded[col].apply(lambda x: encode_input(x))

# Chargement du modèle de classification
with open("covid19_clf.pkl", "rb") as fichier:
    classifieur_charge = pickle.load(fichier)

# Vérifier les noms des caractéristiques attendus par le modèle
print("Noms des caractéristiques attendus par le modèle :", classifieur_charge.feature_names_in_)

# Renommer les colonnes pour correspondre aux noms attendus
df_entree_encoded.columns = classifieur_charge.feature_names_in_

# Affichage des caractéristiques utilisateur
st.subheader("Caractéristiques d'Entrée")
st.dataframe(df_entree)

# Application du modèle pour prédire
prediction = classifieur_charge.predict(df_entree_encoded)
probabilites_prediction = classifieur_charge.predict_proba(df_entree_encoded)

# Résultat de la prédiction
st.subheader("Prédiction")
risque = "Haut risque" if prediction[0] == 1 else "Faible risque"
st.write(f"**{risque}**")

# Probabilités associées à la prédiction
st.subheader("Probabilités de Prédiction")
df_probabilites = pd.DataFrame(probabilites_prediction, columns=["Haut risque", "Faible risque"])
st.dataframe(df_probabilites)
