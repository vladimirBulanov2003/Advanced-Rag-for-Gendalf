
import json
from openai import OpenAI
from numpy import dot
from numpy import argpartition,array
from numpy.linalg import norm
from pydantic import BaseModel, Field
from typing import Optional
from qdrant_client import models
from qdrant_client import QdrantClient
from langchain.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from qdrant_client.models import PayloadSchemaType



def calculate_similarity(query, text):
    return dot(query, text)/(norm(query)*norm(text))

with open("dictionary_with_metadata.json", "r", encoding="utf-8") as f:
    loaded_data_for_metadata = {k: array([text.capitalize() for text in v if text is not None]) for k, v in json.load(f).items()}
    
with open("dictionary_with_embeddings_for_metadata.json", "r", encoding="utf-8") as f:
    loaded_data_for_embeddings = {k: array(v) for k, v in json.load(f).items()}
    

class Metadata(BaseModel):
    product: str =  Field(..., description="Название продукта или модуля, к которому относится информация (например: 1С УПП, Скан-Архив). Желательно опиши это одним или максимум двумя словами!") 
    process: str =  Field(..., description="Описание бизнес-процесса или действия (например: установка, настройка, активация лицензии).") 
    object: str =  Field(..., description="Основной объект или сущность, о которой идет речь (например:  рабочее место, сервер, сканер).")
    error: Optional[str] =  Field(..., description=" Ошибки или проблемы, которые могут возникнуть (например: ошибка подключения, неправильная лицензия).") 
    instruction_type: str =  Field(..., description="Тип инструкции или помощи (например: пошаговая инструкция, описание возможностей, техническая поддержка).")
    system: Optional[str] = Field(..., description=" Система или программная платформа, на которую завязана информация (например: Windows, 1С).") 
    license: Optional[str] = Field(..., description=" Лицензионные аспекты или вопросы активации лицензий (например: добавление лицензии, проверка лицензии).")  


@st.cache_resource
def get_openai_client():
    return OpenAI(api_key = st.secrets["openai"]["api_key"])

def get_embeddings(client, text, model="text-embedding-3-large"):
        return client.embeddings.create(input = text, model=model)

@st.cache_resource
def get_database():
    client_for_qdrant = QdrantClient(url= st.secrets["qdrant"]["url"], 
                                 api_key= st.secrets["qdrant"]["api_key"]
)

    embedding_model = OpenAIEmbeddings(api_key = st.secrets["openai"]["api_key"], model="text-embedding-3-large") 

    vector_store = QdrantVectorStore(
        client=client_for_qdrant,
        collection_name="collection_gendalf",
        embedding=embedding_model,
    )
    
   
    return vector_store
    


def extract_meta_data(client, text, model="gpt-4.1-mini"):
    
    schema =  Metadata.model_json_schema()
    
    promt_for_metadata_extraction = '''
    Ты — интеллектуальный ассистент, помогающий извлекать структурированную информацию из технических пользовательских запросов. 
    На вход ты получаешь один вопрос от пользователя, связанный с настройкой, установкой, лицензированием или интеграцией программных продуктов (например, Скан-Архив, 1С, Windows и др.).

    Задача: Выдели из вопроса значения для следующих полей метаданных:

    Запрос для анализа:  
    <text>
    {text}
    </text>

    Ты обязан отвечать только в JSON формате. Это JSON схема, которую ты обязан соблюдать. Не пиши никаких лишних слов, даже слово json. Название ключа пиши с маленькой буквы, само предложение уже с большой. Пример - product: Скан-Архив
    <schema>
    {metadata_schema}
    </schema>"""
    '''
    
    messages = [{"role": "system", "content": "Ты gpt-4.1-mini, который работает в тех-поддержке компании Гэндальф."}, {"role": "user", "content": promt_for_metadata_extraction.format(text = text, metadata_schema = schema)}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    
    generated_text = response.choices[0].message.content
    try:
        return json.loads(generated_text)
    except json.JSONDecodeError:
        print("Ошибка: Не удалось декодировать сгенерированный текст как JSON.")
        return None
    
    
    
    
def extract_answer_from_llm_using_rag(query_text):
    
    client_for_open_ai = get_openai_client()
    query = get_embeddings(client_for_open_ai, query_text)
    name_of_keys_for_questions = ['first_question', 'second_question', 'third_question']
    max_similarity = -10
    for key in name_of_keys_for_questions:
        for sentence_embedding in loaded_data_for_embeddings[key]:
            similarity = calculate_similarity(query.data[0].embedding, sentence_embedding)
            max_similarity = max(max_similarity, similarity)
    
    if max_similarity < 0.3:
        
        promt_for_inappropriate_question = '''
    
        Пользователь тебе задал этот вопрос: 
        {question}
        
        Данный вопрос, скорее всего, не относится к базе данных.
        Скажи пользователю, что этот вопрос нахолится вне твоей компетенции, и попроси пользователя либо его переформулировать либо действительно помочь.
        Смотри на предедущий контекст и помогай, ТОЛЬКО ЕСЛИ данный вопрос соотносится с тем, что было раньше.
        Например, пользователь можешь попросить что-то дообьяснить, тогда попытайся изложить свои мысли немного по-дургому и наболее простом языке и так далее.
        Если вопрос абсолютно не по теме - просто попроси пользователя вежливо спрашивать про то, что связано с компанией Гэндальф!

        '''
        
        return promt_for_inappropriate_question.format(question = query_text)
            

    meta_data = extract_meta_data(client_for_open_ai, query_text)
    vector_store = get_database()

    k = 10
    filters = []

    for key, value in meta_data.items():
        if value:
            result = [float(calculate_similarity(query.data[0].embedding, text)) for text in loaded_data_for_embeddings[key]]
            indexes = argpartition(result, -k)[-k:]
            best_matches = loaded_data_for_metadata[key][indexes]
            filters.append(models.FieldCondition(key="metadata.{value}".format(value = key),
                    match=models.MatchAny(
                        any = best_matches
                    ),))


    
    results = vector_store.similarity_search(
        query= query_text,
        k=30,
        filter=models.Filter(
            should = filters
        ),
    )


    list_of_similarities_with_query = []
    for name_of_key in name_of_keys_for_questions:
        for doc in results:
            id_of_document = doc.metadata['id']
            path_to_the_folder = doc.metadata['source']
            cos_similarity_with_query = calculate_similarity(query.data[0].embedding, loaded_data_for_embeddings[name_of_key][id_of_document])
            list_of_similarities_with_query.append((id_of_document, cos_similarity_with_query, path_to_the_folder.capitalize()))
        
    list_of_similarities_with_query = sorted(list_of_similarities_with_query, key=lambda element: element[1])

    indexes_of_the_most_relevant_cunks = []
    for tuple in list_of_similarities_with_query[-10:]:
        indexes_of_the_most_relevant_cunks.append(tuple[2])
        
    indexes_of_the_most_relevant_cunks = list(set(indexes_of_the_most_relevant_cunks))

    final_results = vector_store.similarity_search(
        query= query_text,
        k=10,
        filter=models.Filter(
            must = [models.FieldCondition(key="metadata.{value}".format(value = 'source'),
                    match=models.MatchAny(
                        any = indexes_of_the_most_relevant_cunks
                    ),)]
        ),
    )


    PROMPT_TEMPLATE = '''
    Ответь максимально подробно на следующий вопрос:

    {question}

    Используя максимально возможно и качественно следующую информацию, но при этом постарайся включать от туда только то, что необходимо и попытайся отвечать МАКСИМАЛЬНО КРАТКО по пунктам. 
    Так же учти что фрагменты будут перемешаны, однако они могут быть из одинаковых файлов. ОБЯЗАТЕЛЬНО СМОТРИ НА ПУТЬ И НАЗВАНИЯ ФАЙЛА ОТКУДА ПРИШЛИ АБАЗЦЫ И ТЕБЕ НЕОБХОДИМО ИФНОРМАЦИЮ ИЗ ОДИНАКОВЫХ ФАЙЛОВ И ПУТЕЙ ДЕРЖАТЬ ВМЕСТЕ. 
    ТАК ЖЕ ЧРЕЗВЫЧАЙНО ВАЖНО УЧИТЫВАТЬ НАЗВАНИЯ ВСЕХ КНОПОК, ОКОН, ФУНКЦИЙ И ПРОЧЕГО. ПОЭТОМУ ОБЯЗАТЕЛЬНО ПИШИ ИХ НАПРОТИВ КАКОГО-ЛИБО НУЖНОГО ДЕЙСТВИЯ!!!
    ТАК ЖЕ ПРИКЛАДЫВАЙ НАЗВАНИЕ ФАЙЛА ОТКДУДА ТЫ ВЗЯЛ ИНФОРМАЦИЮ!
    СОХРАНЯЙ ТОТ ЖЕ ПОРЯДОК СКРИНШОТОВ ЧТО И В ФРАГМЕНТАХ ТЕКСТА 
    ЕСЛИ ВО ФРАГМЕНТАХ МНОГО СКРИНШОТОВ, ТО ЭТО МОЖЕТ БЫТЬ НОРМОЙ!
    ЕСЛИ В ТВОЕМ ОТВЕТЕ НЕСКОЛЬКО СКРИНШОТОВ ИДУТ ПОДРЯТ - ОСТАВЛЯЙ МЕСТО МЕЖДУ НИМИ ИЛИ КАКИЕ-ТО СИМВОЛЫ, ЧТОБЫ СДЕЛАТЬ ИХ ЧИТАЕМЫМИ!!
    
    
    И СТАРАЙСЯ ОТВЕЧАТЬ ТОЛЬКО ПО ВОПРОСУ КОТОРЫЙ ИНТЕРЕСУЕТ ПОЛЬЗОВАТЕЛЯ БЕЗ КУЧИ ДЕТАЛЕЙ, КОТОРЫЕ НЕ ОТНОСЯТСЯ К САМОМУ ВОПРОСУ!!!
    Вот сам фрагмент текса:

    {context}


    Так же если в информации, которую я скинул выше, есть ссылки на картинки, сайты и прочее, то обязательно их включи РЯДОМ С ТОЙ КОМАНДОЙ, В КОТОРОЙ ОНИ УПОМИНАЮТСЯ!!! НЕ ВКЛЮЧАЙ ИХ В КОНЦЕ, А РЯДОМ С ТЕКСТОМ ГДЕ ЕСТЬ НА НИХ НАМЕК!!!
    
    ОДНАКОЕСЛИ ПОЛЬЗОВАТЕЛЬ ПРОСТО ПРОСИТ ОБЬЯСНИТЬ ПРЕДЫДУЩИЕ ШАГИ ТО ТЫ НЕ ОБЯЗАН ВООБЩЕ ИСПОЛЬЗОВАТЬ САМ ФРАГМЕНТ ДЛЯ ОТВЕТА!!! ИСПОЛЬЗУЙ ФРАГМЕНТ ТОЛЬКО ТАМ, ГДЕ ЭТО НЕОБХОДИМО ДЛЯ ОТВЕТА. ИНОГДА ДОСТАТОЧНО ПОСОМТРЕТЬ НА ПРЕДЫДУЩИЕ ПРЕДЛОЖЕНИЯ!!!
    '''
    context_text = "\n\n---\n\n".join([doc.page_content + "\nПуть из которого взят предыдущий абзац: {source}".format(source = doc.metadata["source"].split("\\")[-1]) for doc in final_results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    return prompt
    
    