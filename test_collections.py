import streamlit as st
from database import Database

def test_collections():
    try:
        db = Database()
        
        st.write("🔍 Vérification des collections :")
        
        # Liste toutes les collections
        collections = db.db.list_collection_names()
        st.write("Collections existantes :", collections)
        
        # Vérifier les index de chaque collection
        for collection in collections:
            st.write(f"\n📚 **Collection : {collection}**")
            
            # Afficher les index
            indexes = list(db.db[collection].list_indexes())
            st.write("Index :")
            for idx in indexes:
                st.write(f"- {idx['name']}: {idx['key']}")
            
            # Afficher le schéma de validation
            try:
                options = db.db.command('listCollections', filter={'name': collection})
                validation = options['cursor']['firstBatch'][0].get('options', {}).get('validator', {})
                if validation:
                    st.write("Schéma de validation :")
                    st.json(validation)
            except Exception as e:
                st.warning(f"Pas de schéma de validation pour {collection}")

    except Exception as e:
        st.error(f"Erreur lors de la vérification : {str(e)}")

def main():
    st.title("🔍 Test des Collections MongoDB")
    
    if st.button("Vérifier les collections"):
        test_collections()

if __name__ == "__main__":
    main() 