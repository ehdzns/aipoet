from dotenv import load_dotenv


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(openai_api_key="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")

from langchain.chat_models import ChatOpenAI


import streamlit as st


st.title('This is a title')
title = st.text_input('Movie title')
result= llm.predict(title)
st.button("Reset", type="primary")
if st.button('Say hello'):
    st.write(result)
else:
    st.write('Goodbye')