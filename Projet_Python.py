import streamlit as st
import pickle
import numpy as np

# Charger le modèle sauvegardé
with open("covid19_clf.pkl", "rb") as file:
    model = pickle.load(file)

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Prédiction COVID-19 - Modèle de Risque",
    page_icon="🩺",
    layout="centered"
)

# Titre principal
st.title("Prédiction du Risque COVID-19")
st.write("""
Bienvenue dans l'outil de prédiction pour évaluer le risque COVID-19 basé sur des facteurs médicaux.
Veuillez remplir les informations ci-dessous pour obtenir une évaluation.
""")

# Liste des variables d'entrée (hors RESULTAT_TEST)
sex = st.selectbox("Sexe :", options=["Femme", "Homme"])
age = st.slider("Âge du patient :", min_value=0, max_value=120, value=30, step=1)
patient_type = st.selectbox("Type de patient :", options=["Non hospitalisé", "Hospitalisé"])
pneumonia = st.selectbox("Pneumonie :", options=["Non", "Oui"])
diabetes = st.selectbox("Diabète :", options=["Non", "Oui"])
copd = st.selectbox("Bronchopneumopathie chronique obstructive (COPD) :", options=["Non", "Oui"])
asthma = st.selectbox("Asthme :", options=["Non", "Oui"])
inmsupr = st.selectbox("Immunodéprimé :", options=["Non", "Oui"])
hipertension = st.selectbox("Hypertension :", options=["Non", "Oui"])
cardiovascular = st.selectbox("Maladies cardiovasculaires :", options=["Non", "Oui"])
renal_chronic = st.selectbox("Maladies rénales chroniques :", options=["Non", "Oui"])
other_disease = st.selectbox("Autres maladies :", options=["Non", "Oui"])
obesity = st.selectbox("Obésité :", options=["Non", "Oui"])
tobacco = st.selectbox("Tabagisme :", options=["Non", "Oui"])
intubed = st.selectbox("Patient intubé :", options=["Non", "Oui"])
icu = st.selectbox("Admission en soins intensifs :", options=["Non", "Oui"])

# Conversion des entrées utilisateur en format numérique pour le modèle
def encode_input(value, positive="Oui", negative="Non"):
    return 1 if value == positive else 2  # "Oui" = 1, "Non" = 2

input_data = np.array([
    1 if sex == "Homme" else 2,  # SEX : Homme = 1, Femme = 2
    age,  # AGE
    1 if patient_type == "Hospitalisé" else 2,  # PATIENT_TYPE : Hospitalisé = 1, Non hospitalisé = 2
    encode_input(pneumonia),  # PNEUMONIA
    encode_input(diabetes),  # DIABETES
    encode_input(copd),  # COPD
    encode_input(asthma),  # ASTHMA
    encode_input(inmsupr),  # INMSUPR
    encode_input(hipertension),  # HIPERTENSION
    encode_input(cardiovascular),  # CARDIOVASCULAR
    encode_input(renal_chronic),  # RENAL_CHRONIC
    encode_input(other_disease),  # OTHER_DISEASE
    encode_input(obesity),  # OBESITY
    encode_input(tobacco),  # TOBACCO
    encode_input(intubed),  # INTUBED
    encode_input(icu)  # ICU
]).reshape(1, -1)

# Bouton pour prédire
if st.button("Prédire le risque"):
    # Effectuer la prédiction
    prediction = model.predict(input_data)[0]  # Prédiction du modèle
    risk = "Haut risque" if prediction == 1 else "Faible risque"
    
    # Afficher le résultat
    st.subheader("Résultat de la prédiction")
    st.success(f"Le patient est classé comme : **{risk}**")
    st.info("Veuillez consulter un professionnel de santé pour une évaluation approfondie.")
