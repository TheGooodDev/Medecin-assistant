import argparse
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from app.utils import load_api_key
from app import config


def ask_question(
    question: str,
    model_name: str,
    temperature: float,
    k: int,
    persist_path: str = "vectorstore",
    user_api_key: str = None  # 👈 AJOUT ICI
) -> None:
    """
    Pose une question à un LLM avec un contexte documentaire vectorisé (RAG).

    Args:
        question (str): La question posée par l'utilisateur.
        model_name (str): Le nom du modèle OpenAI à utiliser.
        temperature (float): Température de génération (plus élevée = plus créatif).
        k (int): Nombre de documents à récupérer via recherche vectorielle.
        persist_path (str): Chemin vers l'index FAISS enregistré.

    Returns:
        None
    """
    # Chargement de la clé API depuis .env
    load_api_key(user_api_key=user_api_key)

    # Chargement de l'index vectoriel (base documentaire)
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.load_local(
        persist_path, embeddings, allow_dangerous_deserialization=True
    )  # 🔥 Obligatoire depuis LangChain 0.1+)

    # Initialisation du modèle LLM OpenAI
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    # Construction de la chaîne RAG (retrieval + question)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": k}),
        return_source_documents=True,
    )

    # Exécution de la requête
    result = qa_chain.invoke({"query": question})

    # Affichage du résultat
    print("🧠 Réponse :\n", result["result"])
    print("\n📄 Sources :")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("source", "Sans source"))

    # ✅ Retourne la réponse (pour l’API ou test)
    return {
        "result": result["result"],
        "source_documents": result["source_documents"]
    }

if __name__ == "__main__":
    # Initialisation du parser de la ligne de commande
    parser = argparse.ArgumentParser(
        description="Poser une question à ton assistant RAG basé sur OpenAI"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=config.DEFAULT_MODEL,
        help="Nom du modèle OpenAI (ex: gpt-3.5-turbo, gpt-4)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.DEFAULT_TEMPERATURE,
        help="Température du modèle (0.0 = déterministe, 1.0 = créatif)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=config.DEFAULT_K,
        help="Nombre de documents pertinents à récupérer",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=False,
        help="Question à poser (sinon posée en interactif)",
    )

    args = parser.parse_args()

    # Récupération de la question (via argument ou input)
    if args.question:
        question = args.question
    else:
        question = input("❓ Pose ta question : ")

    # Exécution principale
    ask_question(
        question=question, model_name=args.model, temperature=args.temperature, k=args.k
    )

