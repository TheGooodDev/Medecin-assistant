"""
Fichier de configuration du projet RAG.

Contient :
- Les paramètres par défaut du modèle
- Les chemins de dossiers
"""

# Modèle et paramètres LLM
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0
DEFAULT_K = 8

# Chemins
DATA_FOLDER = "data"
VECTORSTORE_PATH = "vectorstore"
DATA_FOLDER_AVOCAT = f"{DATA_FOLDER}/loi_marocaine/"
INDEX_TRACKING_FILE = f"{VECTORSTORE_PATH}/indexed_files.json"
