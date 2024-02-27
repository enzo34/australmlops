# Utiliser une image de base Python
FROM python:3.8-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires dans le conteneur
COPY . /app


# Installer les dépendances
RUN pip install --no-cache-dir flask flask-cors joblib scikit-learn==1.3.2 pandas


# Exposer le port sur lequel l'API s'exécute
EXPOSE 5000

# Commande pour exécuter l'API Flask
CMD ["python", "api.py"]
