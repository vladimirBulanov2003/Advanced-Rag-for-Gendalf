
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from main import extract_answer_from_llm_using_rag
from langchain_core.messages import trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import SystemMessage
import uuid


if "graph" not in st.session_state:

    trimmer = trim_messages(strategy="last", max_tokens=5, token_counter=len)
    model = ChatOpenAI(api_key = st.secrets["openai"]["api_key"], model="gpt-4.1-mini")
    workflow = StateGraph(state_schema=MessagesState)

    def call_model(state: MessagesState):
        trimmed_messages = trimmer.invoke(state["messages"])
        system_prompt = (
            "Ты интеллектуальный помощник, который работает в компании Гэндальф. Ты обязан быть вежлив и отвечать на вопросы пользователя."
        )
        messages = [SystemMessage(content=system_prompt)] + trimmed_messages
        response = model.invoke(messages)
        return {"messages": response}
    

    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    

    st.session_state.graph = graph
    st.session_state.memory = memory
    st.session_state.llm = model
    st.session_state.messages = []
    st.session_state.slider = 6




st.title("Тех-поддержка Гэндальф")
st.session_state.slider = st.sidebar.slider("Выставите количество чанков (фрагментов текста), которое будет использоваться для генерации ответа.", 1, 10, st.session_state.slider)

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar='images.png'):
            st.markdown(message.content)


prompt = st.chat_input("Напишите вопрос!")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
        
    promt_for_llm = extract_answer_from_llm_using_rag(prompt, st.session_state.slider)
    result = st.session_state.graph.invoke({"messages": st.session_state.messages + [promt_for_llm]}, config = {"configurable": {"thread_id": "1"}})
    
    with st.chat_message("assistant", avatar='images.png'):
        
        st.markdown(result["messages"][-1].content)
        st.session_state.messages.append(HumanMessage(prompt))
        st.session_state.messages.append(AIMessage(result["messages"][-1].content))
        
  
    
    
    
    
    
    
    
    