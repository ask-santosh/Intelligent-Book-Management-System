from langchain.schema.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from sentence_transformers import SentenceTransformer
import shutil
import time
import torch

from models import Book, SessionLocal

device = 0 if torch.cuda.is_available() else -1  # device=-1 for CPU
print("Using device:", "GPU" if device >= 0 else "CPU")

# Embedding model
MODEL_DIR = "model"  # Path to local embedding model
model_name = "sentence-transformers/all-mpnet-base-v2"  # embedding model

transformer_name, only_model_name = model_name.split("/")
model_loc = os.path.join(MODEL_DIR, transformer_name, only_model_name)
os.makedirs(model_loc, exist_ok=True)

# Path to chroma DB
CHROMA_PATH = "knowledge_library"

# Prompt template for ollama model
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def get_embedding_model():
    global model_name, model_loc

    def is_directory_empty(directory_path):
        return not any(os.scandir(directory_path))

    if is_directory_empty(model_loc):
        from huggingface_hub import login
        login()
        print(f"** Downloading model {model_name}")

        # Load the model from Hugging Face Hub
        model = SentenceTransformer(model_name)

        # Save the model locally
        model.save(model_loc)

        print(f"Model saved to {model_loc}")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_loc,
    )
    return embeddings


# Function to extract text from PDF using pypdf
def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def split_documents(document_list: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(document_list)


def calculate_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def add_knowledge_to_chroma(chunks):
    embedding_model = get_embedding_model()

    # Load the existing database.
    vectordb = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embedding_model
    )

    # Add or Update the documents.
    existing_items = vectordb.get(
        include=[]
    )  # IDs are always included by default
    existing_ids = existing_items["ids"]
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["ids"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        vectordb = Chroma.from_documents(
            embedding=embedding_model,
            persist_directory=CHROMA_PATH,
            ids=new_chunk_ids,
        )
        vectordb.persist()
    else:
        print("✅ No new documents to add")


def clear_database() -> None:
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    print("DB cleanup completed")


def get_words_in_text(text):
    return text.count(" ") + 1


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_model()
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embedding_function
    )

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in results]
    )
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


# Function to summarize a book
def summarize_book(book_title):
    return query_rag(
        f'Please give a summary of the book "{book_title}" in maximum '
        "500 words; Make sure to not provide any title of the summary"
    )


def generate_summary(book_id, book_path, book_title, summary=None):
    start_time = time.time()

    print(book_id, book_path)
    print("generating summary")

    # Extract text from PDF
    print("extracting text from pdf")
    pages = extract_text_from_pdf(book_path)
    print(pages)

    # Chunk extracted texts
    doc_chunks = split_documents(pages)
    print(doc_chunks)

    # Summarize the book
    print("generating summary")
    summary = summarize_book(book_title)

    db = SessionLocal()

    # Fetch the book by ID
    fetched_book = db.query(Book).filter(Book.ids == book_ids).first()
    print(fetched_book.as_dict())

    if fetched_book:
        fetched_book.summary = summary
        db.commit()
        print("Summary added in DB")

    print(f"Whole process took: {time.time() - start_time}")


if __name__ == "__main__":
    pdf_file_path = "uploads/726fe7dc-090a-499f-ae79-982b0124bf25.pdf"
    pdf_book_title = "breast_cancer_nutrition"

    clear_database()

    generate_summary(1, pdf_file_path, pdf_book_title)
