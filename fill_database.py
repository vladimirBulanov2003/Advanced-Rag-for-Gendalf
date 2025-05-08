from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.documents import Document
import json
from langchain.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models
import streamlit as st

with open("documents.json", "r", encoding="utf-8") as f:
     raw_docs = json.load(f)

loaded_docs = [
    Document(
        page_content=doc["page_content"],
        metadata={
            k: v.capitalize() if isinstance(v, str) else v
            for k, v in doc.get("metadata").items()
        }
    )
    for doc in raw_docs
]

embedding_model = OpenAIEmbeddings(api_key = st.secrets["openai"]["api_key"], model="text-embedding-3-large")  

client = QdrantClient(url= st.secrets["qdrant"]["url"], 
                                 api_key= st.secrets["qdrant"]["api_key"])

client.create_collection(
    collection_name="collection_gendalf",
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="collection_gendalf",
    embedding=embedding_model,
)

fields_to_index = ["product", "process", "object", "error", "instruction_type", "system", "license", "integration", "update", "source"]  

for field in fields_to_index:
    client.create_payload_index(
        collection_name="collection_gendalf",
        field_name=f"metadata.{field}",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    
client.create_payload_index(
    collection_name="collection_gendalf",
    field_name="metadata.id",
    field_schema=models.PayloadSchemaType.INTEGER
)

vector_store.add_documents(documents=loaded_docs)

