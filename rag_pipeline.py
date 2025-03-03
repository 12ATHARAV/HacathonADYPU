
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
import torch
from transformers import pipeline

# Step 1: Load the dataset
loader = TextLoader("university_faq.txt")
documents = loader.load()

# Step 2: Split the text into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Step 3: Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Store embeddings in FAISS
vector_db = FAISS.from_documents(texts, embeddings)

# Step 5: Load LLM
model_name = "mistralai/Mistral-7B-v0.1"
pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.float16, device_map="auto")
llm = HuggingFacePipeline(pipeline=pipe)

# Step 6: Create RAG pipeline
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever())

# Step 7: Define and process the query
query = "What is the application deadline for the Computer Science program?"
response = qa_chain.run(query)
print("Response:", response)