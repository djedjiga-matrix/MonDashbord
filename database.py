import pymongo
import pandas as pd
from datetime import datetime
import streamlit as st
import ssl
import certifi

class Database:
    def __init__(self):
        try:
            # Vérifier si les secrets sont disponibles
            if "mongo" not in st.secrets:
                raise Exception("Configuration MongoDB manquante dans .streamlit/secrets.toml")
            
            # Récupérer l'URI
            uri = st.secrets["mongo"]["uri"]
            if not uri:
                raise Exception("URI MongoDB manquant dans la configuration")
            
            # Configuration de la connexion
            self.client = pymongo.MongoClient(
                uri,
                tls=True,
                tlsAllowInvalidCertificates=True,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000
            )
            
            # Tester la connexion
            self.client.admin.command('ping')
            print("✅ Connexion à MongoDB Atlas établie")
            
            self.db = self.client.dashboard_db
            
        except Exception as e:
            print(f"❌ Erreur de connexion à MongoDB: {str(e)}")
            st.error(f"Erreur de connexion à la base de données : {str(e)}")
            # Ne pas lever d'exception, permettre à l'application de continuer
            self.client = None
            self.db = None
    
    def is_connected(self):
        """Vérifie si la connexion à la base de données est établie"""
        return self.db is not None
    
    def save_data(self, data, collection_name):
        try:
            # Supprimer les anciennes données
            self.db[collection_name].delete_many({})
            
            # Copier les données pour éviter de modifier l'original
            data = data.copy()
            
            # Afficher les colonnes disponibles pour le debug
            print(f"Colonnes disponibles dans {collection_name}:", data.columns.tolist())
            
            if collection_name == 'grh_data':
                # Vérifier que les colonnes nécessaires existent
                expected_columns = [
                    'Agents',
                    'Durée production_decimal',
                    'Durée présence_decimal',
                    'Durée pauses_decimal'
                ]
                
                # Vérifier les colonnes manquantes
                missing_columns = [col for col in expected_columns if col not in data.columns]
                if missing_columns:
                    print(f"❌ Colonnes manquantes dans les données GRH : {missing_columns}")
                    return False
                
                # Renommer les colonnes pour MongoDB
                column_mapping = {
                    'Agents': 'agent',
                    'Durée production_decimal': 'duree_production',
                    'Durée présence_decimal': 'duree_presence',
                    'Durée pauses_decimal': 'duree_pauses'
                }
                data = data.rename(columns=column_mapping)
                
                # Convertir les colonnes numériques
                numeric_columns = ['duree_production', 'duree_presence', 'duree_pauses']
                for col in numeric_columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                
                # Remplacer les NaN par 0 pour les colonnes numériques
                data[numeric_columns] = data[numeric_columns].fillna(0)
                
                required_columns = ['agent', 'duree_production', 'duree_presence']
                
            elif collection_name == 'export_data':
                # Vérifier les colonnes requises pour export
                required_columns = [
                    'agent',
                    'contact_qualif1',
                    'contact_qualif2',
                    'pa_montant',
                    'accord_montant',
                    'pour_client',
                    'CMK_S_FIELD_DATETRAITEMENT',
                    'CMK_S_FIELD_NOMFICHIER',
                    'CMK_S_FIELD_DMPROD'
                ]
                # Remplacer les NaN par des valeurs appropriées
                data['pour_client'] = data['pour_client'].fillna('Non spécifié')
                data['contact_qualif1'] = data['contact_qualif1'].fillna('Non spécifié')
                data['contact_qualif2'] = data['contact_qualif2'].fillna('Non spécifié')
                data['agent'] = data['agent'].fillna('Non spécifié')
                # Ajouter la colonne ONG
                data['ONG'] = data['pour_client']
            
            # Vérifier que toutes les colonnes requises sont présentes
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                error_msg = f"Colonnes manquantes dans {collection_name}: {missing_columns}"
                print(error_msg)
                raise ValueError(error_msg)
            
            # Convertir en dictionnaire et sauvegarder
            data_dict = data.to_dict('records')
            for record in data_dict:
                record['date_upload'] = datetime.now()
                # Convertir tous les NaN en None pour MongoDB
                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = None
            
            # Sauvegarder dans MongoDB
            collection = self.db[collection_name]
            collection.insert_many(data_dict)
            
            print(f"✅ Données sauvegardées dans {collection_name}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur détaillée lors de la sauvegarde de {collection_name}: {str(e)}")
            return False
    
    def get_latest_data(self, collection_name):
        try:
            collection = self.db[collection_name]
            data = list(collection.find(
                {}, 
                {'_id': 0}
            ).sort('date_upload', -1))
            
            if data:
                df = pd.DataFrame(data)
                
                if collection_name == 'grh_data':
                    # Renommer les colonnes pour correspondre au format attendu
                    column_mapping = {
                        'agent': 'Agents',
                        'duree_production': 'Durée production_decimal',
                        'duree_presence': 'Durée présence_decimal',
                        'duree_pauses': 'Durée pauses_decimal'
                    }
                    df = df.rename(columns=column_mapping)
                    
                    # Utiliser la colonne Agents au lieu de créer une colonne Equipe
                    print("Colonnes GRH disponibles:", df.columns.tolist())
                    
                elif collection_name == 'export_data':
                    # Créer les colonnes calculées comme dans l'original
                    df['ONG'] = df['pour_client']
                    
                    # Calculer les différents types de qualifications
                    qualif_types = [
                        'don avec montant', 'don en ligne', 'pa en ligne', 
                        'refus argumente', 'pa', 'indecis don'
                    ]
                    
                    for qualif in qualif_types:
                        df[qualif] = df['contact_qualif1'].str.lower().str.contains(qualif, na=False)
                    
                    # Calculer les totaux
                    df['Total Cu'] = (
                        df['refus argumente'].astype(int) + 
                        df['don avec montant'].astype(int) + 
                        df['don en ligne'].astype(int)
                    )
                    df['Total Cu+'] = df['don avec montant'].astype(int) + df['don en ligne'].astype(int)
                    df['Total PA'] = df['pa en ligne'].astype(int)
                    df['Total Indécis'] = df['indecis don'].astype(int)
                    
                    # Calculer les taux
                    df['Tx_Accord_don'] = (df['Total Cu+'] / df['Total Cu'] * 100).round(1)
                    df['Tx_Accord_pa'] = (df['Total PA'] / df['Total Cu'] * 100).round(1)
                    df['Tx_PA_non_converti'] = (df['pa'] / df['pa en ligne'] * 100).round(1)
                    
                    # Convertir les dates
                    df['CMK_S_FIELD_DATETRAITEMENT'] = pd.to_datetime(
                        df['CMK_S_FIELD_DATETRAITEMENT'],
                        format='%Y-%m-%d',
                        errors='coerce'
                    )
                
                return df
            return None
            
        except Exception as e:
            print(f"Erreur lors de la récupération : {str(e)}")
            return None 