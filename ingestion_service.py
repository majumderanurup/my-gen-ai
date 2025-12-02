import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import dotenv_values

class ChromaVectorStore:

    def __init__(self, collection_name="text_data", persist_dir="chroma_store/"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # Load OpenAI API Key
        config = dotenv_values(".env")

        # LangChain embedding model
        self.embedder = OpenAIEmbeddings(
            openai_api_key=config["OPENAI_API_KEY"],
            model="text-embedding-3-small"
        )

        # Native Chroma persistent client
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)

        # Create or load the collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    # ---------------------------------------------------------
    # Recursive Text Splitter
    # ---------------------------------------------------------
    def split_text(self, text, chunk_size=500, chunk_overlap=100):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        )
        return splitter.split_text(text)

    # ---------------------------------------------------------
    # Embed text with OpenAI embeddings
    # ---------------------------------------------------------
    def embed_text(self, texts):
        return self.embedder.embed_documents(texts)

    # ---------------------------------------------------------
    # Store list of large texts with filenames
    # ---------------------------------------------------------
    def store_texts(self, text_list, file_names, chunk_size=500, chunk_overlap=100):
        """
        text_list:   [str, str, str...]
        file_names:  [filename1, filename2, ...] -> must match text_list length
        """

        all_chunks = []
        all_ids = []
        all_metadata = []

        for doc_index, (text, file_name) in enumerate(zip(text_list, file_names)):

            chunks = self.split_text(text, chunk_size, chunk_overlap)

            ids = [
                f"{file_name}_chunk_{i}"
                for i in range(len(chunks))
            ]

            metadata = [
                {"file_name": file_name, "chunk_index": i}
                for i in range(len(chunks))
            ]

            all_chunks.extend(chunks)
            all_ids.extend(ids)
            all_metadata.extend(metadata)

        embeddings = self.embed_text(all_chunks)

        self.collection.add(
            ids=all_ids,
            embeddings=embeddings,
            documents=all_chunks,
            metadatas=all_metadata
        )

        print(f"✅ Stored {len(all_chunks)} chunks from {len(text_list)} files.")

    # ---------------------------------------------------------
    # Retrieve documents + file names
    # ---------------------------------------------------------
    def retrieve(self, query, k=3):
        query_embedding = self.embedder.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        # Chroma returns dict-of-lists, flatten it
        response = []
        for ids, docs, metas in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0]
        ):
            response.append({
                "id": ids,
                "text": docs,
                "file_name": metas.get("file_name"),
                "chunk_index": metas.get("chunk_index")
            })

        return response

    # ---------------------------------------------------------
    # List all items in collection
    # ---------------------------------------------------------
    def list_all(self):
        raw = self.collection.peek()

        items = []
        for ids, docs, metas in zip(
            raw.get("ids", []),
            raw.get("documents", []),
            raw.get("metadatas", [])
        ):
            items.append({
                "id": ids,
                "text": docs,
                "file_name": metas.get("file_name"),
                "chunk_index": metas.get("chunk_index")
            })

        return items

    # ---------------------------------------------------------
    # Delete all chunks belonging to a file
    # ---------------------------------------------------------
    def delete_by_filename(self, file_name):
        # Find IDs for this file
        collection_data = self.collection.get(include=["metadatas"])
        ids_to_delete = []

        for item_id, meta in zip(collection_data["ids"], collection_data["metadatas"]):
            if meta.get("file_name") == file_name:
                ids_to_delete.append(item_id)

        if not ids_to_delete:
            print(f"⚠️ No entries found for file: {file_name}")
            return

        self.collection.delete(ids=ids_to_delete)
        print(f"🗑️ Deleted {len(ids_to_delete)} chunks for file: {file_name}")



# ---------------------------------------------------------
# 1. Create Vector Store
# ---------------------------------------------------------
store = ChromaVectorStore(
    collection_name="demo_collection",
    persist_dir="chroma_store/"
)


# ---------------------------------------------------------
# 2. Sample Texts + Filenames
# ---------------------------------------------------------
texts = [
    """Artificial Intelligence is transforming industries worldwide.
    From healthcare to finance, AI-driven solutions are becoming essential.
    This document explains the basics of modern AI systems.""",

    """Machine Learning is a subset of AI.
    It allows systems to learn from data rather than being explicitly programmed.
    This file discusses supervised and unsupervised learning."""
]

file_names = ["ai_intro.txt", "ml_basics.txt"]


# # ---------------------------------------------------------
# # 3. Store Texts
# # ---------------------------------------------------------
# print("\n📌 Storing documents...\n")
# store.store_texts(texts, file_names)


# # ---------------------------------------------------------
# # 4. Retrieve Similar Chunks
# # ---------------------------------------------------------
# print("\n🔍 Searching for: 'What is AI?'\n")
# results = store.retrieve("What is AI?", k=3)

# for r in results:
#     print(f"➡ ID: {r['id']}")
#     print(f"📄 File: {r['file_name']}")
#     print(f"📎 Chunk: {r['chunk_index']}")
#     print(f"📝 Text: {r['text'][:120]}...\n")


# ---------------------------------------------------------
# 5. List All Chunks
# ---------------------------------------------------------
print("\n📚 List of all stored chunks:\n")
all_items = store.list_all()
for item in all_items:
    print(f"{item['id']} | {item['file_name']} | chunk {item['chunk_index']}")


# ---------------------------------------------------------
# 6. Delete All Chunks of a File
# ---------------------------------------------------------
print("\n🗑️ Deleting file: ml_basics.txt\n")
store.delete_by_filename("ml_basics.txt")


# ---------------------------------------------------------
# 7. List Again to Confirm
# ---------------------------------------------------------
print("\n📚 List after deletion:\n")
all_items_after = store.list_all()
for item in all_items_after:
    print(f"{item['id']} | {item['file_name']} | chunk {item['chunk_index']}")