# PROJET_ML - API Flask avec Docker

## Description

Ce projet contient une application Flask simple, conteneurisée avec Docker. Elle affiche un message de bienvenue à la racine (`/`) et peut être étendue pour inclure du traitement de données ou des prédictions ML.

---

## Structure du projet

examen final farah azer/
│
├── Dockerfile # Instructions Docker pour créer l'image
├── requirements.txt # Liste des dépendances Python (ex: Flask)
├── main.py # Application Flask
├── Online_Retail.csv # Fichier de données (exemple)
└── README.md # Documentation du projet

---
## Architecture du conteneur Docker

- **Image de base** : `python:3.10-slim`
- **Répertoire de travail dans le conteneur** : `/app`
- **Fichiers copiés dans l'image** :
  - `requirements.txt` pour installer les dépendances
  - tout le projet (y compris `main.py`)
- **Commande exécutée** par défaut :
  ```bash
  CMD ["python", "main.py"]
🚀 Instructions d’exécution
1. Construire l’image Docker
bash
Copier
Modifier
docker build -t mon-projet .
2. Exécuter le conteneur
bash
Copier
Modifier
docker run -p 5000:5000 mon-projet

## Résultat attendu
Quand vous accédez à l’URL racine (/), vous voyez :

vbnet
Copier
Modifier
Bienvenue dans l'API Flask avec Docker !

