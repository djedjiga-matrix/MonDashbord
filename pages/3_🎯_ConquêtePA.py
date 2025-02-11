import streamlit as st
import pandas as pd
from utils import display_data_table

st.set_page_config(
    page_title="Conquête PA",
    page_icon="🎯",
    layout="wide"
)

def main():
    st.title("🎯 Conquête Prélèvement Automatique")

    if 'data' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord charger les fichiers sur la page d'accueil.")
        return

    # Récupération des données
    grh_data = st.session_state['data']['grh']
    export_data = st.session_state['data']['export']

    # Filtres dans la sidebar
    with st.sidebar:
        st.header("🔍 Filtres")
        
        statut = st.multiselect(
            "Statut PA",
            options=["En cours", "Validé", "Refusé", "Tous"],
            default="Tous",
            key="statut_filter"
        )
        
        montant = st.slider(
            "Montant PA",
            min_value=0,
            max_value=1000,
            value=(0, 1000),
            key="montant_filter"
        )

    # KPIs spécifiques aux PA
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Nombre de PA",
            value="XXX",
            delta="XX%"
        )
    
    with col2:
        st.metric(
            label="Taux de Validation",
            value="XX%",
            delta="X%"
        )
    
    with col3:
        st.metric(
            label="Montant Total",
            value="XXX €",
            delta="XX%"
        )
    
    with col4:
        st.metric(
            label="PA Moyen",
            value="XX €",
            delta="X%"
        )

    # Graphiques
    tab1, tab2 = st.tabs(["📊 Statuts", "📈 Évolution"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Répartition par Statut")
            # Ajouter le graphique camembert des statuts
            
        with col2:
            st.subheader("Répartition par Montant")
            # Ajouter l'histogramme des montants

    with tab2:
        st.subheader("Évolution des PA")
        # Ajouter le graphique d'évolution temporelle

    # Tableau détaillé des PA
    st.subheader("Détails des PA")
    # Remplacer par votre DataFrame réel
    display_data_table(pd.DataFrame())

if __name__ == "__main__":
    main() 