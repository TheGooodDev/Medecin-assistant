# 🔎 Workflow d'indexation incrémentale RAG avec LangChain + FAISS

Ce document explique le fonctionnement du script `ingest.py` dans un projet RAG (Retrieval-Augmented Generation).

## 🔄 Objectif
Construire ou mettre à jour un index vectoriel FAISS à partir de fichiers PDF dans un dossier `data/`, de manière **incrémentale** (seuls les nouveaux fichiers sont traités).

## ⚙️ Fonctions principales

### `get_all_pdfs(folder_path)`
Retourne la liste des fichiers `.pdf` présents dans un dossier.

### `load_indexed_files()`
Charge la liste des fichiers déjà traités depuis un fichier `indexed_files.json` (dans `vectorstore/`).

### `save_indexed_files(file_list)`
Sauvegarde la liste des fichiers PDF déjà indexés pour éviter de les retraiter.

### `load_documents(file_paths)`
Charge le contenu des fichiers PDF en objets `Document` utilisés par LangChain. Ajoute le nom du fichier dans les métadonnées.

### `build_incremental_vectorstore(...)`
Fonction centrale qui :
- filtre les nouveaux fichiers PDF
- les charge et les découpe en chunks
- crée les embeddings
- met à jour ou crée un index FAISS
- sauvegarde le tout

---

## 📊 Diagramme de la chaîne de traitement

```text
               +-----------------------------+
               |     Dossier "data/"         |
               +-----------------------------+
                     |           |
                     v           v
           get_all_pdfs()   load_indexed_files()
                     \         /
                      \       /
                       \     /
                        v   v
              => Liste des nouveaux fichiers
                        |
                        v
            load_documents() + metadata "source"
                        |
                        v
             CharacterTextSplitter (chunks)
                        |
                        v
           OpenAIEmbeddings (vecteurs numériques)
                        |
                        v
       FAISS: .add_documents() ou .from_documents()
                        |
                        v
         .save_local() + save_indexed_files()
```

---

## 🔹 Résultat
Tu obtiens un index vectoriel dans `vectorstore/` (fichiers `.faiss`, `.pkl` et `.json`), que tu peux interroger dans un pipeline RAG avec LangChain ou autre moteur vectoriel.

