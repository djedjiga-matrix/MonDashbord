import pandas as pd
from io import BytesIO
from datetime import datetime, time
import streamlit as st

def convert_time_to_decimal(time_str):
    """Convertit une durée format 'HH:MM:SS' en heures décimales"""
    try:
        if pd.isna(time_str) or not isinstance(time_str, str):
            return 0.0
        
        hours, minutes, seconds = map(int, time_str.split(':'))
        return hours + minutes/60 + seconds/3600
    except:
        return 0.0

def load_grh_file(content):
    """Charge et traite le fichier GRH en tenant compte des dates"""
    try:
        # Lire le fichier Excel
        excel_file = pd.ExcelFile(BytesIO(content))
        
        # Récupérer les données de présence par date
        presence_data = []
        for sheet_name in excel_file.sheet_names:
            # Ignorer les feuilles spéciales
            if sheet_name in ['Resume', 'Worksheet']:
                continue
                
            try:
                # Vérifier si le nom de la feuille est une date valide
                if not sheet_name[0].isdigit():
                    continue
                    
                sheet_date = pd.to_datetime(sheet_name, format='%Y-%m-%d').date()
                
                # Vérifier si la date est dans la période sélectionnée
                if 'start_date' in st.session_state and 'end_date' in st.session_state:
                    if not (st.session_state['start_date'] <= sheet_date <= st.session_state['end_date']):
                        continue
                
                # Lire le fichier en commençant à la ligne 7 (A7)
                df = pd.read_excel(
                    excel_file, 
                    sheet_name=sheet_name, 
                    header=6,
                    usecols=[0, 12, 22, 23]  # A=0 (Agents), M=12 (Pause), W=22 (Prod), X=23 (Présence)
                )
                
                if not df.empty:
                    # Renommer les colonnes selon leur position
                    df.columns = ['Agents', 'Durée pauses', 'Durée production', 'Durée présence']
                    
                    # Convertir toutes les durées
                    for col in ['Durée production', 'Durée présence', 'Durée pauses']:
                        df[f'{col}_decimal'] = df[col].apply(convert_time_to_decimal)
                    
                    df['Date'] = sheet_date
                    presence_data.append(df)
                    
            except ValueError as ve:
                continue  # Ignorer silencieusement les feuilles non-dates
        
        if presence_data:
            presence_df = pd.concat(presence_data, ignore_index=True)
            result = presence_df.groupby('Agents').agg({
                'Durée production_decimal': 'sum',
                'Durée présence_decimal': 'sum',
                'Durée pauses_decimal': 'sum'
            }).reset_index()
            
            return result
        else:
            st.warning("Aucune donnée de présence trouvée pour la période sélectionnée")
            return None
            
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier GRH : {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def load_export_file(file_content):
    """Charge et traite le fichier d'export"""
    try:
        # Lire le fichier Excel
        excel_data = BytesIO(file_content)
        df = pd.read_excel(excel_data)
        
        # Convertir la colonne date
        if 'CMK_S_FIELD_DATETRAITEMENT' in df.columns:
            df['CMK_S_FIELD_DATETRAITEMENT'] = pd.to_datetime(
                df['CMK_S_FIELD_DATETRAITEMENT'].astype(str),
                format='%Y-%m-%d',
                errors='coerce'
            )
            
            # Filtrer par période si définie
            if 'start_date' in st.session_state and 'end_date' in st.session_state:
                mask = (
                    (df['CMK_S_FIELD_DATETRAITEMENT'].dt.date >= st.session_state['start_date']) & 
                    (df['CMK_S_FIELD_DATETRAITEMENT'].dt.date <= st.session_state['end_date'])
                )
                df = df[mask]
        
        return df
        
    except Exception as e:
        st.error(f"Erreur dans le chargement Export: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None