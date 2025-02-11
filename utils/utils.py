import pandas as pd
from io import BytesIO
from datetime import datetime
import streamlit as st

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

def display_data_table(data):
    """Affiche le tableau de données avec mise en forme"""
    try:
        def highlight_negative(row):
            try:
                is_negative = pd.to_numeric(row['CU\'s à retirer']) < 0
                return ['background-color: #ffcccc' if is_negative else '' for _ in row]
            except:
                return ['' for _ in row]
        
        st.dataframe(
            data.style.apply(highlight_negative, axis=1),
            use_container_width=True,
            hide_index=True
        )
        
        excel_data = to_excel_download(data)
        st.download_button(
            "⬇️ Télécharger (Excel)",
            excel_data,
            f"donnees_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"Erreur d'affichage du tableau : {str(e)}")

colonnes_export = [
    'agent',                          # Nom de l'agent
    'pour_client',                    # ONG/Client
    'contact_qualif1',               # Type de qualification
    'CMK_S_FIELD_DATETRAITEMENT'     # Date de traitement
]

colonnes_grh = [
    'Agents',                        # Nom de l'agent
    'Durée production_decimal',      # Temps de production
    'Durée présence_decimal',        # Temps de présence
    'Durée pauses_decimal'          # Temps de pause
] 