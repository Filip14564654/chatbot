from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Debugging: Print content of some chunks
    for i, chunk in enumerate(chunks[:5]):  # Print first 5 chunks for brevity
        print(f"Chunk {i} content:\n{chunk.page_content[:300]}...\n")
        print(f"Metadata: {chunk.metadata}")

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    try:
        embeddings = OpenAIEmbeddings(api_key="")

        # Generate embeddings for chunks
        chunk_texts = [chunk.page_content for chunk in chunks]
        chunk_embeddings = embeddings.embed_documents(chunk_texts)

        db = Chroma(embedding_function=embeddings, persist_directory=CHROMA_PATH)
        db.add_documents(chunks, embeddings=chunk_embeddings)
        db.persist()
        print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

        # Verify documents are indexed correctly by checking the document count
        verify_indexed_documents(db)
    except Exception as e:
        print(f"Error during saving to Chroma: {e}")


def verify_indexed_documents(db):
    print("Verifying indexed documents...")
    try:
        results = db.similarity_search("", k=1)  # Performing an empty query to get the count
        doc_count = len(results)
        print(f"Number of documents indexed: {doc_count}")
    except Exception as e:
        print(f"Error verifying document count in Chroma: {e}")


if __name__ == "__main__":
    main()
