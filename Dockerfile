# Utiliser une image Python légère (3.11 comme convenu)
FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système pour OpenCV/Pillow
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code et le modèle .pth
COPY . .

# Exposer le port de Streamlit
EXPOSE 8501

# Lancer l'app avec Docker qui surveille le processus
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]