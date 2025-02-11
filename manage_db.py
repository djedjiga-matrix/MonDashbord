from database import Database
import pandas as pd

def init_local_db():
    db = Database()
    db.init_db()
    
    # Charger les fichiers locaux
    try:
        grh_data = pd.read_excel('data/extract_grh.xlsx')
        export_data = pd.read_excel('data/export.xlsx')
        
        # Sauvegarder dans la base
        db.save_data(grh_data, 'grh_data')
        db.save_data(export_data, 'export_data')
        print("✅ Données chargées avec succès dans la base locale")
    except Exception as e:
        print(f"❌ Erreur : {str(e)}")

if __name__ == "__main__":
    init_local_db() 