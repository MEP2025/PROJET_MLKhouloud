import pandas as pd
import numpy as np
from io import StringIO
import json
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Configuration des chemins
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, 'artifacts', 'model_artifacts.joblib')

# Variables globales avec valeurs par défaut
MODEL = None
SCALER = None
TFIDF = None
COUNTRIES = ['United Kingdom']  # Valeur par défaut
FEATURE_COLUMNS = ['Quantity', 'TotalPrice']  # Colonnes de base

class DummyModel:
    """Modèle factice complet avec tous les composants"""
    def __init__(self):
        # Vocabulaire factice
        self.dummy_vocab = ['heart', 'light', 'holder', 'white', 'hanging']
        
        # TF-IDF ajusté
        self.tfidf = TfidfVectorizer(max_features=100)
        self.tfidf.fit([' '.join(self.dummy_vocab)])
        
        # Scaler ajusté
        self.scaler = StandardScaler()
        # Ajustement avec des données factices
        dummy_data = np.array([[1, 10.0], [2, 20.0]])  # Exemple de données
        self.scaler.fit(dummy_data)
        
        # Pays supportés
        self.countries = ['United Kingdom', 'France', 'Germany']
        
        # Colonnes attendues
        self.feature_columns = ['Quantity', 'TotalPrice'] + [
            f'tfidf_{i}' for i in range(100)
        ] + [
            f'Country_{country}' for country in self.countries
        ]
    
    def predict(self, X):
        return np.array([15.0] * len(X))  # Prix moyen factice

def load_artifacts():
    """Charge les artefacts du modèle avec gestion robuste"""
    global MODEL, SCALER, TFIDF, COUNTRIES, FEATURE_COLUMNS
    
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"⚠ Attention: Fichier modèle non trouvé à {MODEL_PATH}")
            print("Mode développement activé avec modèle factice complet")
            MODEL = DummyModel()
            SCALER = MODEL.scaler
            TFIDF = MODEL.tfidf
            COUNTRIES = MODEL.countries
            FEATURE_COLUMNS = MODEL.feature_columns
            return False
        
        artifacts = joblib.load(MODEL_PATH)
        MODEL = artifacts['model']
        SCALER = artifacts['scaler']
        TFIDF = artifacts['tfidf_vectorizer']
        
        # Vérifications des artefacts
        if not hasattr(TFIDF, 'vocabulary_'):
            raise ValueError("TF-IDF non ajusté")
        if not hasattr(SCALER, 'scale_'):
            raise ValueError("Scaler non ajusté")
            
        COUNTRIES = artifacts.get('countries', ['United Kingdom'])
        FEATURE_COLUMNS = artifacts.get('feature_columns', [])
        
        print("✔ Modèle chargé avec succès")
        return True
        
    except Exception as e:
        print(f" Erreur de chargement du modèle: {str(e)}")
        print("Utilisation du modèle factice complet")
        MODEL = DummyModel()
        SCALER = MODEL.scaler
        TFIDF = MODEL.tfidf
        COUNTRIES = MODEL.countries
        FEATURE_COLUMNS = MODEL.feature_columns
        return False

def preprocess_data(df):
    """Prétraitement robuste avec gestion des cas limites"""
    # Vérification des entrées
    if not isinstance(df, pd.DataFrame):
        raise ValueError("L'entrée doit être un DataFrame pandas")
    
    # Gestion des colonnes manquantes
    df['UnitPrice'] = df.get('UnitPrice', 1.0)
    df['Quantity'] = df.get('Quantity', 1)
    df['Description'] = df.get('Description', 'unknown').fillna('unknown')
    df['Country'] = df.get('Country', 'United Kingdom').fillna('United Kingdom')
    
    # Calcul des features de base
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    # Vectorisation TF-IDF
    try:
        tfidf_matrix = TFIDF.transform(df['Description'])
        tfidf_cols = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols)
    except Exception as e:
        raise ValueError(f"Erreur TF-IDF: {str(e)}")
    
    # Encodage des pays
    country_dummies = pd.get_dummies(df['Country'], prefix='Country')
    # S'assure que tous les pays attendus sont présents
    for country in COUNTRIES:
        col_name = f'Country_{country}'
        if col_name not in country_dummies:
            country_dummies[col_name] = 0
    
    # Construction du dataset final
    features = pd.concat([
        tfidf_df,
        df[['Quantity', 'TotalPrice']],
        country_dummies[[f'Country_{c}' for c in COUNTRIES]]
    ], axis=1)
    
    # Ajout des colonnes manquantes
    for col in FEATURE_COLUMNS:
        if col not in features:
            features[col] = 0
    
    # Sélection des colonnes dans le bon ordre
    try:
        features = features[FEATURE_COLUMNS]
    except KeyError as e:
        missing = set(FEATURE_COLUMNS) - set(features.columns)
        raise ValueError(f"Colonnes manquantes: {missing}")
    
    # Normalisation - seulement si le scaler est ajusté
    if SCALER is not None and hasattr(SCALER, 'transform'):
        return SCALER.transform(features)
    return features.values

def input_fn(input_data, content_type):
    """Convertit les données d'entrée en DataFrame"""
    if content_type == 'application/json':
        data = json.loads(input_data)
        if isinstance(data, dict):
            return pd.DataFrame([data])
        return pd.DataFrame(data)
    elif content_type == 'text/csv':
        return pd.read_csv(StringIO(input_data))
    raise ValueError(f"Type non supporté: {content_type}")

def predict_fn(input_data, model):
    """Effectue la prédiction"""
    processed_data = preprocess_data(input_data)
    return model.predict(processed_data)

def output_fn(prediction, accept):
    """Formate la sortie"""
    if accept == 'application/json':
        return json.dumps({'predictions': prediction.tolist()})
    raise ValueError(f"Format de sortie non supporté: {accept}")

# Initialisation
model_loaded = load_artifacts()


from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()
        input_df = input_fn(json.dumps(input_data), "application/json")
        prediction = predict_fn(input_df, MODEL)
        result = output_fn(prediction, "application/json")
        return jsonify(json.loads(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Test avec différents cas
    test_cases = [
        {
            "Description": "WHITE HANGING HEART T-LIGHT HOLDER",
            "Quantity": 6,
            "UnitPrice": 2.55,
            "Country": "United Kingdom"
        },
        {
            "Description": "RED WOOLLY HOTTIE WHITE HEART",
            "Quantity": 3,
            "UnitPrice": 3.39,
            "Country": "France"
        }
    ]
    
    print("\n🔍 Démarrage des tests...")
    for i, test_data in enumerate(test_cases, 1):
        try:
            print(f"\nTest {i}: {test_data['Description']}")
            input_df = input_fn(json.dumps(test_data), "application/json")
            processed = preprocess_data(input_df)
            print(f"✔ Données prétraitées ({processed.shape})")
            
            prediction = predict_fn(input_df, MODEL)
            result = output_fn(prediction, "application/json")
            
            print("✅ Prédiction réussie:")
            print(json.dumps(json.loads(result), indent=2))
            
        except Exception as e:
            print(f" Échec du test: {str(e)}")
    
    if not model_loaded:
        print("\nℹ Pour utiliser le vrai modèle:")
        print("1. Créez un dossier 'artifacts'")
        print("2. Exécutez train_model.py pour générer model_artifacts.joblib")
        print("3. Placez le fichier dans le dossier artifacts")
        print("Contenu attendu du fichier:")
        print("- 'model': modèle scikit-learn entraîné")
        print("- 'scaler': StandardScaler ajusté")
        print("- 'tfidf_vectorizer': TfidfVectorizer ajusté")
        print("- 'countries': liste des pays supportés")
        print("- 'feature_columns': liste des colonnes attendues")