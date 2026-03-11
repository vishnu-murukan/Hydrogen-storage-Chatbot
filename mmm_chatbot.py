import os
from dotenv import load_dotenv
import pinecone
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from docx import Document
import re
import numpy as np

# === Load API Keys from .env ===
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

try:
    pinecone_client = pinecone.Pinecone(api_key=pinecone_api_key)
    genai.configure(api_key=google_api_key)
except Exception as e:
    print(f"API initialization error: {str(e)}")
    exit(1)

index_name = "scopus-corpus-index"
dimension = 768
try:
    if index_name not in pinecone_client.list_indexes().names():
        pinecone_client.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine"
        )
    index = pinecone_client.Index(index_name)
except pinecone.exceptions.PineconeException as pe:
    print(f"Pinecone error: {str(pe)}")
    exit(1)

embed_model = SentenceTransformer('allenai-specter')

def preprocess_text(text):
    text = re.sub(r"© \d{4}.*?All rights reserved\.", "", text)
    text = re.sub(r"Graphical Abstract:.*?\)", "", text)
    return text.strip()

def load_corpus(file_path):
    try:
        doc = Document(file_path)
        corpus = []
        for i, para in enumerate(doc.paragraphs):
            text = preprocess_text(para.text.strip())
            if text:
                corpus.append({"title": f"Doc_{i}", "abstract": text})
        print(f"Extracted {len(corpus)} documents.")
        return corpus
    except Exception as e:
        print(f"Error loading corpus: {str(e)}")
        return []

def chunk_text_with_overlap(text, max_chars=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start += max_chars - overlap
    return chunks

def get_embedding(text):
    if not text.strip():
        return [0.0] * 768
    return embed_model.encode(text).tolist()

def store_in_pinecone(corpus):
    vectors = []
    for i, doc in enumerate(corpus):
        text = doc.get("title", "") + " " + doc.get("abstract", "")
        if text.strip():
            chunks = chunk_text_with_overlap(text)
            for j, chunk in enumerate(chunks):
                embedding = get_embedding(chunk)
                vector_id = f"{i}_chunk_{j}"
                metadata = {"title": doc.get("title", ""), "abstract_chunk": chunk}
                vectors.append((vector_id, embedding, metadata))
                print(f"Prepared vector {vector_id}")
    if vectors:
        try:
            index.upsert(vectors=vectors)
            print(f"Stored {len(vectors)} vectors.")
        except pinecone.exceptions.PineconeException as pe:
            print(f"Error upserting vectors: {str(pe)}")
    else:
        print("No vectors to store.")
    stats = index.describe_index_stats()
    print(f"Index stats: {stats}")

def safe_get_metadata(match, key, default="Unknown"):
    return match['metadata'].get(key, default)

def retrieve_and_generate(query, top_k=3):
    query_embedding = get_embedding(query)
    try:
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        context_parts = []
        for i, match in enumerate(results['matches'], 1):
            title = safe_get_metadata(match, 'title')
            abstract = safe_get_metadata(match, 'abstract_chunk')
            score = match['score']
            print(f"Match {i}: Title={title}, Score={score}")
            print(f"Abstract: {abstract[:200]}...")
            context_parts.append(f"Document {i}:\nTitle: {title}\nAbstract: {abstract}")
        context = "\n\n".join(context_parts)
        if not context_parts:
            return "No relevant documents found for the query."

        model = genai.GenerativeModel("models/gemini-1.5-pro")
        response = model.generate_content(
            f"You are an expert in hydrogen storage materials. Based on these documents, don't go out of topic:\n\n{context}\n\n"
            f"Question: {query}\n"
            "Provide a detailed answer based on the documents or give a relevant response."
        )
        return response.text
    except Exception as e:
        return f"Error during retrieval or generation: {str(e)}"

def main():
    corpus_file = "corpus.docx"  # Replace with actual file
    corpus = load_corpus(corpus_file)
    if not corpus:
        print("Failed to load corpus.")
        return
    store_in_pinecone(corpus)
    print("\nEnter your query (type 'stop' to exit):")
    while True:
        query = input("> ").strip()
        if query.lower() == "stop":
            print("Exiting.")
            break
        if not query:
            print("Please enter a valid query.")
            continue
        answer = retrieve_and_generate(query)
        print(f"Answer: {answer}\n")

if __name__ == "__main__":
    main()
