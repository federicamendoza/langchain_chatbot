import os
import pymysql
from dotenv import load_dotenv
import re
from sqlalchemy import create_engine, text
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
from langdetect import detect
from bs4 import BeautifulSoup
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough

def clean_html(raw_html: str) -> str:
    return BeautifulSoup(raw_html, "html.parser").get_text(separator=" ", strip=True)


def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang
    except:
        return "it"  # default fallback

# 1. Carica le variabili d'ambiente (anche se per Ollama non sono strettamente necessarie)
load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = os.getenv("MYSQL_PORT")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

# --- Funzioni di connessione e estrazione dati da MySQL ---
def get_mysql_data(table_name: str, query: str = None) -> list[str]:
    connection_string = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
    )
    engine = create_engine(connection_string)
    documents = []

    try:
        with engine.connect() as connection:
            if query:
                result = connection.execute(text(query))
            else:
                result = connection.execute(text(f"SELECT * FROM {table_name}"))

            for row in result:
                row_dict = row._mapping
                course_name = row_dict.get("titolo", "").lower()

                # --- Estrazione Livello e Aggiornamento dal titolo ---
                course_level = None
                is_update = False

                # Livello
                if "base" in course_name or "basso" in course_name:
                    course_level = "base"
                elif "intermedio" in course_name or "medio" in course_name:
                    course_level = "intermedio"
                elif "avanzato" in course_name or "alto" in course_name:
                    course_level = "avanzato"

                # Aggiornamento
                if "aggiornamento" in course_name:
                    is_update = True

                descrizione = clean_html(row_dict.get("descrizione", ""))
                requisiti = clean_html(row_dict.get("requisiti", ""))

                page_content = ", ".join([
                    f"titolo: {row_dict.get('titolo', '')}",
                    f"descrizione: {descrizione}",
                    f"requisiti: {requisiti}",
                    f"COSTO DEL CORSO: {row_dict.get('MaxCostoFormat', '')}",
                    f"ore: {row_dict.get('ore', '')}",
                    f"datasito: {row_dict.get('datasito', '')}"
                ])
                metadata = {
                    "nome_corso_originale": row_dict.get("nome_corso"),
                    "data_corso": str(row_dict.get("data_corso")),
                    "livello": course_level,
                    "aggiornamento": is_update,
                    "id_corso": row_dict.get("id"),
                }

                documents.append(Document(page_content=page_content, metadata=metadata))

        print(f"Dati estratti e metadati parsati con successo dalla tabella {table_name}.")
        return documents

    except Exception as e:
        print(f"Errore durante l'estrazione dati da MySQL: {e}")
        return []

# --- Configurazione del modello di embedding e del Vector Store ---
def setup_vector_store(documents: list[Document], persist_directory: str = "./chroma_db") -> Chroma:
    embeddings = OllamaEmbeddings(model="llama2")

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("Caricamento del Vector Store esistente...")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print("Creazione di un nuovo Vector Store...")
        vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)

    print("Vector Store pronto.")
    return vectorstore


# --- Configurazione dell'LLM e della catena RAG ---
def setup_rag_chain(vectorstore: Chroma) -> RetrievalQA:
    llm = Ollama(model="mistral", temperature=0.1)

    qa_template = """Sei un assistente utile e preciso per informazioni sui corsi. Rispondi alla domanda basandoti SOLO ed esclusivamente sul contesto fornito.
    Se la domanda è ambigua o ci sono più corsi con titoli simili, chiedi all'utente di specificare meglio il nome, il livello o la data del corso.

    Rispondi sempre nella stessa lingua della domanda dell'utente.

    Quando ci sono più versioni di un corso nel contesto, segui queste priorità:
    - Se la domanda non specifica una data, dai precedenza alla versione più recente del corso.
    - Se la domanda non specifica un livello (es. base, intermedio, avanzato), includi informazioni su tutti i livelli disponibili per quel corso.
    - Se la domanda non specifica gli aggiornamenti, includi anche informazioni su eventuali aggiornamenti del corso.

    Se la risposta non può essere trovata nel contesto fornito, rispondi che non hai informazioni sufficienti su quel corso o su quella specifica versione.
    Non inventare risposte.

    Contesto: {context}
    Domanda: {query}
    Risposta utile:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        template=qa_template,
        input_variables=["context", "query"]
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # This is a more modern and explicit way to create the RAG chain using LCEL
    # It avoids the ambiguity of the legacy RetrievalQA.from_chain_type
    rag_chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | QA_CHAIN_PROMPT
        | llm
    )

    return rag_chain

def main():
    my_query = """SELECT
    c.titolo,
    COALESCE(c.descrizione, 'nessuna descrizione') AS descrizione,
    COALESCE(c.requisiti, 'nessuno') AS requisiti,
    c.testocosto AS MaxCostoFormat,
    c.ore,
    c.datasito AS UltimoDATASITO
FROM corsi c
JOIN (
        SELECT
            titolo,
            MAX(
                CASE
                    WHEN testocosto = '(richiedere)' THEN 0
                    WHEN testocosto REGEXP '^[0-9]+' THEN CAST(REGEXP_SUBSTR(testocosto, '^[0-9]+') AS UNSIGNED)
                    ELSE 0
                END
                ) AS max_costo
        FROM corsi
        WHERE evidenza_id = 2
        GROUP BY titolo 
    ) AS max_costi ON
    max_costi.titolo = c.titolo AND
    (
        (c.testocosto = '(richiedere)' AND 0 = max_costi.max_costo) OR
        (c.testocosto REGEXP '^[0-9]+' AND CAST(REGEXP_SUBSTR(c.testocosto, '^[0-9]+') AS UNSIGNED) = max_costi.max_costo)
    )
    ORDER BY
    c.datasito DESC;"""
    
    mysql_data_strings = get_mysql_data(table_name="corsi", query=my_query)

    # Debug: print all indexed course titles
    for doc in mysql_data_strings:
        print("INDEXED COURSE:", doc.page_content)

    if not mysql_data_strings:
        print("Nessun dato estratto dal database MySQL. Impossibile procedere.")
        return

    # 2. Setup del Vector Store con OllamaEmbeddings
    vectorstore = setup_vector_store(mysql_data_strings)

    # 3. Setup della catena RAG con Ollama LLM
    rag_chain = setup_rag_chain(vectorstore)

    print("\nCiao! Sono il tuo bot per assisterti per informazioni sui corsi. Digita la tua domanda (o 'esc' per uscire).")

    while True:
        query = input("\nLa tua domanda: ")
        if query.lower() == 'esc':
            break

        try:
            # lang = detect_language(query) # No longer needed for the prompt
            print(f"DEBUG: Query being sent: {query}")
            # The input to an LCEL chain is the query string directly
            retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
            retrieved_docs = retriever.invoke(query)
            print("RETRIEVED DOCS FOR QUERY:", [doc.page_content for doc in retrieved_docs])
            result = rag_chain.invoke(query)

            print("\nRisposta:", result)
        except Exception as e:
            print(f"Si è verificato un errore durante la generazione della risposta: {e}")

if __name__ == "__main__":
    main()
