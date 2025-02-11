import streamlit as st
import pandas as pd
from utils import display_data_table

st.set_page_config(
    page_title="Performance Globale",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("📊 Performance Globale")

    # Vérifier si les données sont chargées
    if 'data' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord charger les fichiers sur la page d'accueil.")
        return

    # Récupération des données
    grh_data = st.session_state['data']['grh']
    export_data = st.session_state['data']['export']

    # Filtres dans la sidebar
    with st.sidebar:
        st.header("🔍 Filtres")
        date_range = st.date_input(
            "Période",
            value=(pd.Timestamp.now() - pd.Timedelta(days=30), pd.Timestamp.now()),
            key="date_range"
        )
        
        if grh_data is not None:
            # Filtrer par agent
            agent = st.selectbox(
                "Filtrer par agent",
                options=['Tous'] + sorted(grh_data['Agents'].unique().tolist()),
                key="agent_filter"
            )

    # Filtrer les données selon l'agent sélectionné
    if agent != 'Tous' and grh_data is not None:
        filtered_grh = grh_data[grh_data['Agents'] == agent]
        filtered_export = export_data[export_data['agent'] == agent]
    else:
        filtered_grh = grh_data
        filtered_export = export_data

    # Affichage des KPIs principaux
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_appels = len(filtered_export) if filtered_export is not None else 0
        st.metric(
            label="Total Appels",
            value=f"{total_appels:,}",
            delta=None
        )
    
    with col2:
        if filtered_export is not None:
            tx_conversion = (
                filtered_export['Total Cu+'].sum() / 
                filtered_export['Total Cu'].sum() * 100 
                if filtered_export['Total Cu'].sum() > 0 else 0
            )
        else:
            tx_conversion = 0
        st.metric(
            label="Taux de Conversion",
            value=f"{tx_conversion:.1f}%",
            delta=None
        )
    
    with col3:
        if filtered_export is not None:
            temps_moyen = filtered_export['CMK_S_FIELD_DMPROD'].mean() / 60
        else:
            temps_moyen = 0
        st.metric(
            label="Temps Moyen d'Appel",
            value=f"{temps_moyen:.1f} min",
            delta=None
        )
    
    with col4:
        if filtered_grh is not None:
            performance = filtered_grh['Durée production_decimal'].mean()
        else:
            performance = 0
        st.metric(
            label="Performance Globale",
            value=f"{performance:.1f}%",
            delta=None
        )

    # Graphiques
    tab1, tab2 = st.tabs(["📈 Évolution", "📊 Répartition"])
    
    with tab1:
        st.subheader("Évolution des performances")
        # Ajouter le graphique d'évolution ici
        st.line_chart(pd.DataFrame())  # À remplacer par vos données réelles

    with tab2:
        st.subheader("Répartition par équipe")
        # Ajouter le graphique de répartition ici
        st.bar_chart(pd.DataFrame())  # À remplacer par vos données réelles

    # Tableau détaillé
    st.subheader("Détails des performances")
    # Remplacer par votre DataFrame réel
    display_data_table(pd.DataFrame())

if __name__ == "__main__":
    main() 