from openai import OpenAI
from numpy import dot
from numpy import argmax, argpartition,array
from numpy.linalg import norm
import json
from collections import defaultdict
import streamlit as st

client = OpenAI(api_key = st.secrets["openai"]["api_key"])

def get_embeddings(text, model="text-embedding-3-large"):
    return client.embeddings.create(input = text, model=model)

with open("dictionary_with_metadata.json", "r", encoding="utf-8") as f:
    loaded_data = {k: [text for text in v if text is not None] for k, v in json.load(f).items()}

dict_with_embeddings = defaultdict(list)
    
for name_of_meta_group, list_of_data in loaded_data.items():
    dict_with_embeddings[name_of_meta_group] =  [embed.embedding for embed in get_embeddings(list_of_data).data]
    
with open("dictionary_with_embeddings_for_metadata.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in dict_with_embeddings.items()}, f, ensure_ascii=False)
       
