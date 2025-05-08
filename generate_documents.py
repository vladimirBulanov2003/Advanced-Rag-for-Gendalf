import asyncio

import openai
from typing import Optional
from pydantic import BaseModel, Field
import json
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import defaultdict
from langchain_core.documents import Document
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import streamlit as st
class Metadata(BaseModel):
    product: Optional[str] =  Field(..., description="Название продукта или модуля, к которому относится информация (например: 1С УПП, Скан-Архив). Желательно опиши это одним или максимум двумя словами!") 
    process: Optional[str] =  Field(..., description="Описание бизнес-процесса или действия (например: установка, настройка, активация лицензии).") 
    object: Optional[str] =  Field(..., description="Основной объект или сущность, о которой идет речь (например:  рабочее место, сервер, сканер).")
    error: Optional[str] =  Field(..., description=" Ошибки или проблемы, которые могут возникнуть (например: ошибка подключения, неправильная лицензия).") 
    instruction_type: str =  Field(..., description="Тип инструкции или помощи (например: пошаговая инструкция, описание возможностей, техническая поддержка).")
    system: Optional[str] = Field(..., description=" Система или программная платформа, на которую завязана информация (например: Windows, 1С).") 
    license: Optional[str] = Field(..., description=" Лицензионные аспекты или вопросы активации лицензий (например: добавление лицензии, проверка лицензии).")  
    integration: Optional[str] = Field(..., description=" Интеграции с другими системами или ПО (например: интеграция с 1С, синхронизация со сканером).")   
    update: Optional[str] = Field(..., description=" Описание обновлений, патчей, новых версий продукта (например: обновление Скан-Архива).")   
    
    
    
class Hypothetical(BaseModel):
    first_question: str =  Field(..., description= "Первый вопрос, отражающий ключевую тему чанка.") 
    second_question: str =  Field(..., description="Второй вопрос, фокусирующийся на деталях или примерах.") 
    third_question: str =  Field(..., description="Третий вопрос, связанный с применением или следствиями.")
     

promt_for_metadata_extraction = """Вычлени всю информацию (meta данные) из этого куска текста.

            Текст:
            <text>
            {text}
            </text>
            Ты обязан отвечать только в JSON формате. Это JSON схема, которую ты обязан соблюдать. Не пиши никаких лишних слов, даже слово json. ВСЕГДА ВСЕ META ДАННЫЕ ПИШИ С ЗАГЛАВНОЙ БУКВЫ!! Но именно названия ключей оставляй с маленькой (например : process : Выгрузка документов):
            <schema>
            {metadata_schema}
            </schema>"""

promt_for_hypothetical_questions = """Ваша задача — сгенерировать 3 гипотетических вопроса на основе следующего текста. Вопросы должны:

            1. Быть краткими и конкретными.
            2. Отражать ключевые аспекты текста.
            3. Покрывать разные уровни детализации (общая тема → детали → применение).
            
            Вот как примерно могут выглядеть задаваемые вопросы: 
            
            1) У нас уже активировано 1 рабочее место, и мы купили лицензию на дополнительное место, как его активировать?
            2) Мы работаем в 1С Управление производственным предприятием, как нам установить Скан-Архив?
            3) Для каких конфигураций подходит Скан-Архив?
            4) Есть ли возможность распознавать документы в Скан-Архиве без штрихкода или со штрихкодом от 1С?
            
            Текст для анализа:  
            <text>
            {text}
            </text>
            Ты обязан отвечать только в JSON формате. Это JSON схема, которую ты обязан соблюдать. Не пиши никаких лишних слов, даже слово json. Название ключа пиши с маленькой буквы, само предложение уже с большой. Пример - first_question: Что умеет делать модуль 1C?
            <schema>
            {metadata_schema}
            </schema>"""
            
schema_for_metadata =  Metadata.model_json_schema()
schema_for_hypothetical_questions = Hypothetical.model_json_schema()



client = openai.AsyncOpenAI(api_key = st.secrets["openai"]["api_key"])


async def make_api_call_to_gpt(text, schema, user_prompt,  model="gpt-4.1-mini"):
   
    messages = [{"role": "system", "content": "Ты gpt-4.1-mini, который работает в тех-поддержке компании Гэндальф."}, {"role": "user", "content": user_prompt.format(text = text, metadata_schema = schema)}]
    response = await client.chat.completions.create(
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


async def main():
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.abspath(os.path.join(base_dir, "", "annotations"))


    loader = DirectoryLoader(
        path=docs_dir,
        glob="**/*.docx",
        loader_cls=UnstructuredWordDocumentLoader,
    )

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2048,
        chunk_overlap = 210,
        separators=["\n## ", "\n### ", "\n\n"]
    )

    texts = text_splitter.split_documents(documents)
    
    dict_with_data = {'product' : set(), 
                      'process' : set(), 
                      'object': set(),  
                      'error' : set(),  
                      'instruction_type': set(),  
                      'system': set(),  
                      'license' : set(), 
                      'integration' : set(),  
                      'update' : set(), 
                      'first_question' : [],
                      'second_question': [],
                      'third_question': []}
    
    results_for_metadata = []
    results_for_hypothetical_questions = []
    array_with_documents = []
    id_for_document = 0
    offset = 60
    
    for index in range(0, len(texts), offset):
    
        tasks_for_metadata = [make_api_call_to_gpt(text.page_content, schema_for_metadata, promt_for_metadata_extraction) for text in texts[index: index + offset]]
        tasks_for_hypothetical_questions = [make_api_call_to_gpt(text.page_content, schema_for_hypothetical_questions, promt_for_hypothetical_questions) for text in texts[index: index + offset]]

        results_for_metadata = await asyncio.gather(*tasks_for_metadata)
        results_for_hypothetical_questions = await asyncio.gather(*tasks_for_hypothetical_questions)
        
        for output_for_metadata, output_for_questions, doc in zip(results_for_metadata, results_for_hypothetical_questions, texts[index: index + offset]):
            for key, value in output_for_metadata.items():
                key = key.lower()
                if value:
                 value = value.capitalize()
                dict_with_data[key].add(value)
                
            for key, value in output_for_questions.items():
                key = key.lower()
                dict_with_data[key].append(value)
            
            text = doc.page_content
            text = re.sub(r'[\n\t\xa0]+', ' ', text).strip()
            text = re.sub(r'\s+', ' ', text)
            
            output_for_metadata['source'] = doc.metadata['source']
            output_for_metadata['id'] = id_for_document
            
            array_with_documents.append(Document(page_content = text,  metadata = output_for_metadata))
            id_for_document += 1
                
    
    with open("documents.json", "w", encoding="utf-8") as f:
        json.dump([doc.dict() for doc in array_with_documents], f, ensure_ascii=False)
          
    with open("dictionary_with_metadata.json", "w", encoding="utf-8") as f:
        json.dump({k: list(v) for k, v in dict_with_data.items()}, f, ensure_ascii=False)
            
asyncio.run(main())


