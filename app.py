import streamlit as st
from rag_pipeline import qa_chain

st.title("University Admissions FAQ Chatbot")

# Input for user query
query = st.text_input("Ask a question:")

# Process query and display response
if query:
    response = qa_chain.run(query)
    st.write("Response:", response)