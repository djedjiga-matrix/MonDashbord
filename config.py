import streamlit as st

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