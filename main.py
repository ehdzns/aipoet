from langchain.chat_models import ChatOpenAI

import streamlit as st

llm = ChatOpenAI(openai_api_key= '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')






st.title('This is a title')
title = st.text_input('Movie title')
result= llm.predict(title)
st.button("Reset", type="primary")
if st.button('Say hello'):
    st.write(result)
else:
    st.write('Goodbye')


