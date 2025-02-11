import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from data_loader import load_grh_file, load_export_file
from io import BytesIO
from plotly.subplots import make_subplots
import numpy as np
from config import initialize_session_state
from database import Database
from utils.utils import to_excel_download, display_data_table

def initialize_session_state():
    """Initialise les états de session pour les filtres"""
    if 'conquest_filters' not in st.session_state:
        st.session_state['conquest_filters'] = {
            'ong': "Toutes",
            'percentage': '7%',
            'type': 'PPA'
        }
    
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "Performance Globale"

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Centre d'Appels",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialiser l'état de session
initialize_session_state()

# Initialiser la base de données
db = Database()

# Vérifier la connexion
if not db.is_connected():
    st.warning("⚠️ Mode hors ligne : les données ne seront pas sauvegardées")
else:
    st.success("✅ Connecté à la base de données")

def to_excel_download(df):
    """Convertit le DataFrame en fichier Excel téléchargeable"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Performance', index=False)
        worksheet = writer.sheets['Performance']
        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).apply(len).max(),
                len(str(col))
            ) + 2
            worksheet.set_column(idx, idx, max_length)
    output.seek(0)
    return output.read()

def create_agent_table(export_df, grh_df, start_date=None, end_date=None):
    """Crée le tableau de performance des agents"""
    try:
        # Créer le DataFrame de base
        agent_stats = pd.DataFrame()
        
        # Calculer les statistiques par agent
        if export_df is not None:
            # Copier le DataFrame pour éviter les modifications sur l'original
            export_df = export_df.copy()
            
            # Convertir et filtrer les dates
            if 'CMK_S_FIELD_DATETRAITEMENT' in export_df.columns:
                # Convertir la colonne en datetime avec le bon format
                export_df['CMK_S_FIELD_DATETRAITEMENT'] = pd.to_datetime(
                    export_df['CMK_S_FIELD_DATETRAITEMENT'],
                    format='%Y-%m-%d',  # Format modifié pour YYYY-MM-DD
                    errors='coerce'
                )
                
                # Obtenir les dates min et max en filtrant les NaT
                valid_dates = export_df['CMK_S_FIELD_DATETRAITEMENT'].dropna()
                
                if not valid_dates.empty:
                    min_date = valid_dates.min().date()
                    max_date = valid_dates.max().date()
                    
                    # Sélecteur de dates
                    dates = st.date_input(
                        "Sélectionner la période",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        key="date_range_create"
                    )
                    
                    if isinstance(dates, tuple) and len(dates) == 2:
                        start_date, end_date = dates
                    else:
                        start_date, end_date = min_date, max_date
                else:
                    st.warning("Aucune date valide trouvée dans les données")
                    start_date = end_date = None
            
            # Obtenir l'ONG par agent
            agent_ong = export_df.groupby('agent')['pour_client'].agg(
                lambda x: x.mode()[0] if not x.empty else 'N/A'
            ).reset_index()
            agent_ong.columns = ['agent', 'ONG']
            
            # Créer le DataFrame de base avec agent et ONG
            agent_stats = agent_ong
            
            # Compter les différents types de qualifications
            for qualif in [
                'don avec montant', 'don en ligne', 'pa en ligne', 'refus argumente',
                'pa', 'indecis don'
            ]:
                agent_stats[qualif] = agent_stats['agent'].apply(
                    lambda x: export_df[
                        export_df['agent'] == x
                    ]['contact_qualif1'].str.lower().str.contains(qualif, na=False).sum()
                )
            
            # Calculer Total CU et CU+
            agent_stats['Total Cu'] = (
                agent_stats['refus argumente'] + 
                agent_stats['don avec montant'] + 
                agent_stats['don en ligne']
            )
            agent_stats['Total Cu+'] = agent_stats['don avec montant'] + agent_stats['don en ligne']
            agent_stats['Total PA'] = agent_stats['pa en ligne']  # Uniquement PA en ligne
            agent_stats['Total Indécis'] = agent_stats['indecis don']
            
            # Calculer les taux
            agent_stats['Tx_Accord_don'] = (agent_stats['Total Cu+'] / agent_stats['Total Cu'] * 100).round(1)
            agent_stats['Tx_Accord_pa'] = (agent_stats['Total PA'] / agent_stats['Total Cu'] * 100).round(1)
            agent_stats['Tx_PA_non_converti'] = (agent_stats['pa'] / agent_stats['pa en ligne'] * 100).round(1)  # Nouveau taux
            
            # Initialiser les colonnes pour la conquête
            agent_stats['CU\'s à facturer'] = 0
            agent_stats['CU\'s à retirer'] = 0
            
        # Ajouter les données de temps si disponibles
        if grh_df is not None:
            temps_data = grh_df.groupby('Agents').agg({
                'Durée production_decimal': 'sum',
                'Durée présence_decimal': 'sum',
                'Durée pauses_decimal': 'sum'
            }).reset_index()
            
            temps_data.columns = ['agent', 'Total H/prod', 'Total/H presence', 'Total/H Brief']
            
            # Fusionner avec les stats agents
            agent_stats = agent_stats.merge(temps_data, on='agent', how='left')
            
            # Calculer CU's/h
            agent_stats['Cu\'s/h'] = (agent_stats['Total Cu'] / agent_stats['Total H/prod']).round(2)
        
        # Formater les colonnes numériques
        numeric_cols = agent_stats.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if 'Tx_' in col:
                agent_stats[col] = agent_stats[col].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "0%")
            elif '/h' in col:
                agent_stats[col] = agent_stats[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "0")
            elif 'H/' in col or '/H' in col:
                agent_stats[col] = agent_stats[col].apply(lambda x: f"{x:.2f}h" if pd.notnull(x) else "0h")
            else:
                agent_stats[col] = agent_stats[col].apply(lambda x: f"{int(x)}" if pd.notnull(x) else "0")
        
        # Vérifier si le DataFrame est vide avant de le retourner
        if agent_stats.empty:
            st.warning("Aucune donnée trouvée pour la période sélectionnée")
            # Créer un DataFrame avec les colonnes attendues
            return pd.DataFrame(columns=[
                'ONG', 'agent', 'don avec montant', 'don en ligne', 'pa en ligne',
                'refus argumente', 'Total Cu', 'Total Cu+', 'Tx_Accord_don',
                'Tx_Accord_pa', 'Cu\'s/h', 'Total H/prod', 'Total/H presence',
                'Total/H Brief'
            ])
        
        return agent_stats
    except Exception as e:
        st.error(f"Erreur lors de la création du tableau : {str(e)}")
        # Retourner un DataFrame vide avec les colonnes attendues
        return pd.DataFrame(columns=[
            'ONG', 'agent', 'don avec montant', 'don en ligne', 'pa en ligne',
            'refus argumente', 'Total Cu', 'Total Cu+', 'Tx_Accord_don',
            'Tx_Accord_pa', 'Cu\'s/h', 'Total H/prod', 'Total/H presence',
            'Total/H Brief'
        ])

def calculate_conquest_metrics(df, ong_selected, percentage, conquest_type):
    """Calcule les métriques de conquête"""
    try:
        if df is None or df.empty:
            return pd.DataFrame()
            
        # Créer une copie du DataFrame
        result_df = df.copy()
        
        # Debug: afficher les colonnes disponibles
        print("Colonnes disponibles:", result_df.columns.tolist())
        
        # Filtrer par ONG si sélectionnée
        if ong_selected != "Toutes":
            result_df = result_df[result_df['ONG'] == ong_selected]
        
        # Convertir le pourcentage en décimal
        perc = float(percentage.rstrip('%')) / 100
        
        # Debug: afficher les premières lignes avant conversion
        print("\nDonnées avant conversion:")
        print(result_df[['pa en ligne', 'Total Cu+', 'Total Cu']].head())
        
        # Convertir les colonnes nécessaires en numérique
        for col in ['pa en ligne', 'Total Cu+', 'Total Cu']:
            result_df[col] = pd.to_numeric(result_df[col].str.replace(',', ''), errors='coerce').fillna(0)
        
        # Debug: afficher les premières lignes après conversion
        print("\nDonnées après conversion:")
        print(result_df[['pa en ligne', 'Total Cu+', 'Total Cu']].head())
        
        # Calculer CU's à facturer selon le type de conquête
        if conquest_type == 'PPA':
            # Pour PPA: pa en ligne / pourcentage
            result_df['CU\'s à facturer'] = result_df['pa en ligne'].apply(
                lambda x: int(x / perc) if x > 0 else 0
            )
        else:  # PRP
            # Pour PRP: Total Cu+ / pourcentage
            result_df['CU\'s à facturer'] = result_df['Total Cu+'].apply(
                lambda x: int(x / perc) if x > 0 else 0
            )
        
        # Calculer CU's à retirer
        result_df['CU\'s à retirer'] = (result_df['CU\'s à facturer'] - result_df['Total Cu']).astype(int)
        
        # Debug: afficher le résultat final
        print("\nRésultat final:")
        print(result_df[['CU\'s à facturer', 'CU\'s à retirer']].head())
        
        return result_df
        
    except Exception as e:
        print(f"Erreur détaillée: {str(e)}")
        st.error(f"Erreur lors du calcul des métriques : {str(e)}")
        return pd.DataFrame()

def create_performance_heatmap(data, metric_type='cu'):
    """
    Crée une carte de chaleur des performances
    metric_type: 'cu' (Contacts Utiles), 'cu_plus' (Contacts Positifs), 
                'pa' (PA en ligne), 'taux_accord' (Taux d'accord)
    """
    try:
        # Convertir la colonne date en datetime
        data['CMK_S_FIELD_DATETRAITEMENT'] = pd.to_datetime(data['CMK_S_FIELD_DATETRAITEMENT'])
        
        # Extraire l'heure
        data['heure'] = data['CMK_S_FIELD_DATETRAITEMENT'].dt.hour
        
        # Définir les conditions selon le type de métrique
        if metric_type == 'cu':
            condition = data['contact_qualif1'].str.lower().isin(['refus argumente', 'don avec montant', 'don en ligne'])
            title = 'Contacts Utiles par heure et par agent'
        elif metric_type == 'cu_plus':
            condition = data['contact_qualif1'].str.lower().isin(['don avec montant', 'don en ligne'])
            title = 'Contacts Positifs par heure et par agent'
        elif metric_type == 'pa':
            condition = data['contact_qualif1'].str.lower() == 'pa en ligne'
            title = 'PA en ligne par heure et par agent'
        elif metric_type == 'taux_accord':
            # Pour le taux d'accord, nous devons calculer différemment
            cu = data.groupby(['agent', 'heure'])['contact_qualif1'].apply(
                lambda x: x.str.lower().isin(['refus argumente', 'don avec montant', 'don en ligne']).sum()
            ).reset_index(name='cu')
            
            cu_plus = data.groupby(['agent', 'heure'])['contact_qualif1'].apply(
                lambda x: x.str.lower().isin(['don avec montant', 'don en ligne']).sum()
            ).reset_index(name='cu_plus')
            
            # Fusionner et calculer le taux
            pivot_data = cu.merge(cu_plus, on=['agent', 'heure'])
            pivot_data['taux'] = (pivot_data['cu_plus'] / pivot_data['cu'] * 100).fillna(0)
            
            # Créer le pivot pour la heatmap
            heatmap_data = pivot_data.pivot(
                index='agent',
                columns='heure',
                values='taux'
            ).fillna(0)
            
            title = "Taux d'accord par heure et par agent (%)"
            
            # Créer la figure pour le taux d'accord
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='RdYlBu',  # Rouge pour faible, bleu pour élevé
                colorbar=dict(title='%')
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Heure de la journée',
                yaxis_title='Agent',
                height=400
            )
            
            return fig
            
        # Pour les autres métriques
        data_filtered = data[condition].copy()
        heatmap_data = pd.pivot_table(
            data_filtered,
            values='contact_qualif1',
            index='agent',
            columns='heure',
            aggfunc='count',
            fill_value=0
        )
        
        # Créer la figure pour les autres métriques
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Blues',
            colorbar=dict(title='Nombre')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Heure de la journée',
            yaxis_title='Agent',
            height=400
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Erreur lors de la création de la heatmap : {str(e)}")
        return None

def create_conversion_funnel(data):
    """Crée un diagramme en entonnoir du processus de conversion"""
    try:
        # Calculer les étapes du funnel
        funnel_data = {
            'Contacts Totaux': len(data),
            'Contacts Utiles': data['Total Cu'].astype(int).sum(),
            'Contacts Positifs': data['Total Cu+'].astype(int).sum(),
            'PA en ligne': data['pa en ligne'].astype(int).sum()
        }
        
        # Créer le funnel
        fig = go.Figure(go.Funnel(
            y=list(funnel_data.keys()),
            x=list(funnel_data.values()),
            textinfo="value+percent initial"
        ))
        
        fig.update_layout(
            title='Entonnoir de Conversion',
            height=400
        )
        
        return fig
    except Exception as e:
        st.error(f"Erreur lors de la création du funnel : {str(e)}")
        return None

def create_performance_trends(data, date_col='CMK_S_FIELD_DATETRAITEMENT'):
    """Crée un graphique des tendances de performance"""
    try:
        # Grouper par date et contact_qualif1
        data[date_col] = pd.to_datetime(data[date_col])
        
        # Créer les métriques par jour
        daily_stats = data.groupby([data[date_col].dt.date, 'contact_qualif1']).size().reset_index(name='count')
        
        # Calculer les différentes métriques
        daily_cu = daily_stats[
            daily_stats['contact_qualif1'].str.contains('refus argumente|don avec montant|don en ligne', 
                                                      case=False, na=False)
        ].groupby(date_col)['count'].sum().reset_index()
        daily_cu.columns = [date_col, 'Contacts Utiles']
        
        daily_cup = daily_stats[
            daily_stats['contact_qualif1'].str.contains('don avec montant|don en ligne', 
                                                      case=False, na=False)
        ].groupby(date_col)['count'].sum().reset_index()
        daily_cup.columns = [date_col, 'Contacts Positifs']
        
        daily_pa = daily_stats[
            daily_stats['contact_qualif1'].str.contains('pa en ligne', 
                                                      case=False, na=False)
        ].groupby(date_col)['count'].sum().reset_index()
        daily_pa.columns = [date_col, 'PA en ligne']
        
        # Fusionner les données
        merged_data = daily_cu.merge(daily_cup, on=date_col, how='outer').merge(daily_pa, on=date_col, how='outer')
        merged_data = merged_data.fillna(0)
        
        # Créer le graphique
        fig = go.Figure()
        
        # Ajouter les lignes pour chaque métrique
        metrics = {
            'Contacts Utiles': 'Contacts Utiles',
            'Contacts Positifs': 'Contacts Positifs',
            'PA en ligne': 'PA en ligne'
        }
        
        for col, name in metrics.items():
            fig.add_trace(go.Scatter(
                x=merged_data[date_col],
                y=merged_data[col],
                name=name,
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title='Évolution des performances',
            xaxis_title='Date',
            yaxis_title='Nombre de contacts',
            height=400,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    except Exception as e:
        st.error(f"Erreur lors de la création des tendances : {str(e)}")
        return None

def detect_anomalies(data, metric, threshold=2):
    """Détecte les anomalies dans les performances"""
    try:
        # Convertir la métrique en numérique
        values = pd.to_numeric(data[metric], errors='coerce')
        
        # Calculer les statistiques
        mean = values.mean()
        std = values.std()
        
        # Identifier les anomalies (valeurs au-delà de threshold écarts-types)
        anomalies = data[abs(values - mean) > threshold * std]
        
        return anomalies
    except Exception as e:
        st.error(f"Erreur lors de la détection des anomalies : {str(e)}")
        return pd.DataFrame()

def show_conquest_don_page(agent_stats):
    """Page dédiée à la Conquête Don"""
    st.title("🎯 Conquête Don")
    
    # Filtres spécifiques pour Conquête Don
    col1, col2, col3 = st.columns(3)
    with col1:
        ong_list = ["Toutes"] + sorted(agent_stats['ONG'].unique().tolist())
        ong_selected = st.selectbox(
            "Sélectionner l'ONG", 
            ong_list, 
            key="don_ong"
        )
    with col2:
        percentage = st.selectbox(
            "Pourcentage", 
            ['1%', '4%', '6%', '7%', '7.5%', '12%'], 
            key="don_perc"
        )
    with col3:
        conquest_type = st.selectbox(
            "Type de Don",
            ['PRP', 'PPA'],
            key="don_type"
        )
    
    # Calcul des métriques Don
    conquest_data = calculate_conquest_metrics(agent_stats, ong_selected, percentage, conquest_type)
    
    if not conquest_data.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Tableau des données")
            display_data_table(conquest_data)
        
        with col2:
            st.subheader("📊 Métriques Globales")
            
            # Calcul des métriques globales
            total_cu = pd.to_numeric(conquest_data['Total Cu']).sum()
            total_cu_facturer = pd.to_numeric(conquest_data['CU\'s à facturer']).sum()
            tx_accord_don = pd.to_numeric(conquest_data['Tx_Accord_don'].str.rstrip('%')).mean()
            cus_h = pd.to_numeric(conquest_data['Cu\'s/h'].str.replace(',', '')).mean()
            
            # Affichage des métriques
            st.metric("Taux d'accord Don Global", f"{tx_accord_don:.1f}%")
            st.metric("Total CU Global", f"{total_cu:,.0f}")
            st.metric("CU's à facturer Global", f"{total_cu_facturer:,.0f}")
            st.metric("CU/h Global", f"{cus_h:.2f}")
    else:
        st.warning("Aucune donnée trouvée pour les filtres sélectionnés")

def show_conquest_pa_page(agent_stats):
    """Page dédiée à la Conquête PA"""
    st.title("🎯 Conquête PA")
    
    # Filtres spécifiques pour Conquête PA
    col1, col2, col3 = st.columns(3)
    with col1:
        ong_list = ["Toutes"] + sorted(agent_stats['ONG'].unique().tolist())
        ong_selected = st.selectbox(
            "Sélectionner l'ONG", 
            ong_list, 
            key="pa_ong"
        )
    with col2:
        percentage = st.selectbox(
            "Pourcentage", 
            ['1%', '4%', '6%', '7%', '7.5%', '12%'], 
            key="pa_perc"
        )
    with col3:
        conquest_type = st.selectbox(
            "Type de PA",
            ['PPA', 'PRP'],
            key="pa_type"
        )
    
    # Calcul des métriques PA
    conquest_data = calculate_conquest_metrics(agent_stats, ong_selected, percentage, conquest_type)
    
    if not conquest_data.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Tableau des données")
            display_data_table(conquest_data)
        
        with col2:
            st.subheader("📊 Métriques Globales")
            
            # Calcul des métriques globales
            total_pa_ligne = pd.to_numeric(conquest_data['pa en ligne']).sum()
            total_pa = pd.to_numeric(conquest_data['Total PA']).sum()
            tx_accord_pa = pd.to_numeric(conquest_data['Tx_Accord_pa'].str.rstrip('%')).mean()
            
            # Calculer la moyenne de PA par jour (en utilisant le total d'heures de production)
            total_heures = pd.to_numeric(conquest_data['Total H/prod'].str.rstrip('h')).sum()
            nb_jours = total_heures / 8 if total_heures > 0 else 1  # Supposant 8h par jour
            moy_pa_jour = total_pa_ligne / nb_jours if nb_jours > 0 else 0
            
            # Taux de PA non convertis
            tx_pa_non_converti = pd.to_numeric(conquest_data['Tx_PA_non_converti'].str.rstrip('%')).mean()
            
            # Affichage des métriques
            st.metric("Total PA en ligne Global", f"{total_pa_ligne:,.0f}")
            st.metric("Total PA Global", f"{total_pa:,.0f}")
            st.metric("Taux d'accord PA Global", f"{tx_accord_pa:.1f}%")
            st.metric("Moyenne PA/jour", f"{moy_pa_jour:.1f}")
            st.metric("Taux PA non convertis Global", f"{tx_pa_non_converti:.1f}%")
    else:
        st.warning("Aucune donnée trouvée pour les filtres sélectionnés")

def display_data_table(data):
    """Affiche le tableau de données avec mise en forme"""
    try:
        # Styling pour les lignes avec CU's à retirer négatif
        def highlight_negative(row):
            try:
                is_negative = pd.to_numeric(row['CU\'s à retirer']) < 0
                return ['background-color: #ffcccc' if is_negative else '' for _ in row]
            except:
                return ['' for _ in row]
        
        # Afficher le tableau
        st.dataframe(
            data.style.apply(highlight_negative, axis=1),
            use_container_width=True,
            hide_index=True
        )
        
        # Export Excel
        excel_data = to_excel_download(data)
        st.download_button(
            "⬇️ Télécharger (Excel)",
            excel_data,
            f"donnees_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"download_table_{datetime.now().strftime('%H%M%S')}"
        )
    except Exception as e:
        st.error(f"Erreur d'affichage du tableau : {str(e)}")

# Interface utilisateur
st.title("📊 Dashboard Centre d'Appels")

# Navigation principale
page = st.sidebar.selectbox(
    "Navigation",
    ["Performance Globale", "Conquête Don", "Conquête PA", "Graphiques"],
    key="nav_main"
)

# Sidebar avec filtres
with st.sidebar:
    st.header("📁 Fichiers")
    grh_file = st.file_uploader(
        "Fichier GRH (extract_grh_...)", 
        type=["xlsx"],
        key="grh_file_upload"
    )
    export_file = st.file_uploader(
        "Fichier Export (export_...)", 
        type=["xlsx"],
        key="export_file_upload"
    )
    
    # Ajout des filtres globaux
    if 'data' in st.session_state:
        st.header("🔍 Filtres")
        export_df = st.session_state['data']['export']
        
        # Afficher les colonnes disponibles pour le debug
        st.write("Colonnes disponibles:", export_df.columns.tolist())
        
        # Vérifier quelle colonne utiliser pour les ONGs
        ong_columns = ['ONG', 'pour_client', 'client', 'organisation']
        ong_column = next((col for col in ong_columns if col in export_df.columns), None)
        
        if ong_column:
            # Convertir toutes les valeurs en string avant le tri
            ong_values = [str(x) for x in export_df[ong_column].unique() if pd.notna(x)]
            ong_list = ["Toutes"] + sorted(ong_values)
            
            st.selectbox("ONG", ong_list, key="global_ong")
            st.selectbox("Pourcentage", ['1%', '4%', '6%', '7%', '7.5%', '12%'], key="global_percentage")
            st.selectbox("Type de campagne", ['PRP', 'PPA'], key="global_type")
        else:
            st.warning("⚠️ Aucune colonne d'ONG trouvée dans les données")

# Après le chargement des données
if grh_file is not None and export_file is not None:
    try:
        st.info("🔄 Chargement des fichiers en cours...")
        grh_data = load_grh_file(grh_file.read())
        export_data = load_export_file(export_file.read())
        
        # Sauvegarder les données dans MongoDB
        if db.save_data(grh_data, 'grh_data'):
            if db.save_data(export_data, 'export_data'):
                # Stocker dans la session Streamlit
                st.session_state['data'] = {
                    'grh': grh_data,
                    'export': export_data
                }
                st.success("✅ Données chargées et sauvegardées avec succès!")
            else:
                st.error("❌ Erreur lors de la sauvegarde des données d'export")
        else:
            st.error("❌ Erreur lors de la sauvegarde des données GRH")
            
    except Exception as e:
        st.error(f"❌ Une erreur s'est produite : {str(e)}")
else:
    # Essayer de charger les dernières données depuis MongoDB
    grh_data = db.get_latest_data('grh_data')
    export_data = db.get_latest_data('export_data')
    
    if grh_data is not None and export_data is not None:
        st.session_state['data'] = {
            'grh': grh_data,
            'export': export_data
        }
        st.info("📊 Données chargées depuis la base de données")
    else:
        st.info("👈 Veuillez charger les fichiers GRH et Export dans la barre latérale")