# Image de base légère Python 3.11
FROM python:3.11-slim

# Répertoire de travail
WORKDIR /app

# Installer uniquement les dépendances système nécessaires pour Streamlit, Pillow/OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Mettre à jour pip
RUN pip install --upgrade pip

# Copier les fichiers de dépendances Python
COPY requirements.txt .

# Installer les dépendances Python sans cache
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application
COPY . .

# Exposer le port Streamlit
EXPOSE 8501

# Lancer l'application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
