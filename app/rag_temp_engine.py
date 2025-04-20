# app/rag_temp_engine.py

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.utils import load_api_key

def run_rag_on_uploaded_docs(
    docs,
    question: str,
    model: str,
    temperature: float,
    k: int,
    user_api_key: str = None
) -> dict:
    """
    Applique la logique RAG en mémoire sur des documents temporairement uploadés.

    Args:
        docs (List[Document]): Liste de documents LangChain
        question (str): Question à poser
        model (str): Nom du modèle OpenAI
        temperature (float): Température du modèle
        k (int): Top K vecteurs
        user_api_key (str, optional): Clé API perso, sinon .env ou st.secrets

    Returns:
        dict: résultat du RAG avec 'result' et 'source_documents'
    """
    load_api_key(user_api_key=user_api_key)

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(chunks, embeddings)

    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model_name=model, temperature=temperature)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    return qa_chain.invoke({"query": question})
