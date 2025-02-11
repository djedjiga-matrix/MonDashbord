import streamlit as st
import pandas as pd
from utils import display_data_table

st.set_page_config(
    page_title="Conquête Don",
    page_icon="🎯",
    layout="wide"
)

def main():
    st.title("🎯 Conquête Don")

    if 'data' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord charger les fichiers sur la page d'accueil.")
        return

    # Récupération des données
    grh_data = st.session_state['data']['grh']
    export_data = st.session_state['data']['export']

    # Filtres dans la sidebar
    with st.sidebar:
        st.header("🔍 Filtres")
        
        ong = st.selectbox(
            "ONG",
            options=["Toutes"] + list(export_data['ONG'].unique()),
            key="ong_filter"
        )
        
        percentage = st.select_slider(
            "Pourcentage de don",
            options=['5%', '7%', '10%', 'Tous'],
            value='7%',
            key="percentage_filter"
        )

    # KPIs spécifiques aux dons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Nombre de Dons",
            value="XXX",
            delta="XX%"
        )
    
    with col2:
        st.metric(
            label="Montant Total",
            value="XXX €",
            delta="XX%"
        )
    
    with col3:
        st.metric(
            label="Don Moyen",
            value="XX €",
            delta="X%"
        )

    # Graphiques
    tab1, tab2, tab3 = st.tabs(["📊 Répartition", "📈 Évolution", "🎯 Objectifs"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Répartition par ONG")
            # Ajouter le graphique camembert ici
            
        with col2:
            st.subheader("Répartition par Montant")
            # Ajouter le graphique barres ici

    with tab2:
        st.subheader("Évolution des dons")
        # Ajouter le graphique d'évolution temporelle
        
    with tab3:
        st.subheader("Suivi des objectifs")
        # Ajouter les indicateurs de progression

    # Tableau détaillé des dons
    st.subheader("Détails des dons")
    # Remplacer par votre DataFrame réel
    display_data_table(pd.DataFrame())

if __name__ == "__main__":
    main() 