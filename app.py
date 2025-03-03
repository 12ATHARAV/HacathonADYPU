import streamlit as st
from rag_pipeline import qa_chain

# Set up the Streamlit app
st.set_page_config(page_title="University Admissions FAQ Chatbot", page_icon="🎓")

# Add a title and description
st.title("🎓 University Admissions FAQ Chatbot")
st.write("Welcome! Ask me anything about university admissions, deadlines, scholarships, and more.")

# Input for user query
query = st.text_input("Ask a question:")

# Process query and display response
if query:
    with st.spinner("Thinking..."):
        response = qa_chain.run(query)
        st.success("Response:")
        st.write(response)
