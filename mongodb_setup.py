# mongodb_setup.py
import streamlit as st
from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime
from database import Database

def setup_mongodb_collections():
    """Configure les collections MongoDB avec leurs index et validations"""
    try:
        db = Database()
        
        # 1. Collection pour les données d'export
        export_collection = db.db['export_data']
        export_collection.create_index([("date_traitement", ASCENDING)])
        export_collection.create_index([("agent", ASCENDING)])
        export_collection.create_index([("pour_client", ASCENDING)])
        
        # 2. Collection pour les données GRH
        grh_collection = db.db['grh_data']
        grh_collection.create_index([("date", ASCENDING)])
        grh_collection.create_index([("agent", ASCENDING)])
        
        # 3. Collection pour les performances
        performance_collection = db.db['performance_metrics']
        performance_collection.create_index([
            ("date", ASCENDING),
            ("agent", ASCENDING)
        ])
        
        # Définir les règles de validation
        db.db.command({
            "collMod": "export_data",
            "validator": {
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["agent", "contact_qualif1", "pour_client"],
                    "properties": {
                        "agent": {"bsonType": "string"},
                        "contact_qualif1": {"bsonType": "string"},
                        "pour_client": {"bsonType": "string"}
                    }
                }
            }
        })
        
        db.db.command({
            "collMod": "grh_data",
            "validator": {
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["agent", "duree_production", "duree_presence"],
                    "properties": {
                        "agent": {"bsonType": "string"},
                        "duree_production": {"bsonType": "double"},
                        "duree_presence": {"bsonType": "double"}
                    }
                }
            }
        })
        
        st.success("✅ Collections MongoDB configurées avec succès!")
        
        # Afficher la structure
        st.write("📚 Structure de la base de données:")
        collections = {
            "export_data": "Données des appels",
            "grh_data": "Données RH",
            "performance_metrics": "Métriques de performance"
        }
        
        for coll, desc in collections.items():
            indexes = db.db[coll].list_indexes()
            st.write(f"\n**{coll}**: {desc}")
            st.write("Index créés:")
            for idx in indexes:
                st.write(f"- {idx['name']}: {idx['key']}")
                
    except Exception as e:
        st.error(f"❌ Erreur lors de la configuration : {str(e)}")

def main():
    st.title("🗄️ Configuration MongoDB")
    st.write("""
    Cette page permet de configurer la structure de la base de données MongoDB 
    pour l'application du centre d'appels.
    """)
    
    if st.button("🚀 Configurer les collections"):
        setup_mongodb_collections()

if __name__ == "__main__":
    main()