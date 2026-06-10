import os
import streamlit as st
import json
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Zyro Dynamics HR Portal", page_icon="🤖", layout="wide")
st.title("🤖 Zyro Dynamics HR Help Desk")
st.markdown("---")

@st.cache_resource
def initialize_rag_system():
    corpus_path = "."

    loader = PyPDFDirectoryLoader(corpus_path)
    documents = loader.load()

    if not documents:
        st.error("No PDF documents found! Please make sure your 11 policy PDFs are uploaded directly to the main page of your GitHub repository.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 25, "lambda_mult": 0.5}
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=512)
    return retriever, llm

if "GROQ_API_KEY" not in os.environ:
    st.sidebar.subheader("Configuration Panel")
    user_key = st.sidebar.text_input("Enter Groq API Key:", type="password")
    if user_key:
        os.environ["GROQ_API_KEY"] = user_key
    else:
        st.warning("🔒 System Standby: Please provide your Groq API Key in the sidebar to load the portal.")
        st.stop()

try:
    retriever, llm = initialize_rag_system()
except Exception as e:
    st.error(f"System Error while generating vector indexes: {str(e)}")
    st.stop()

REFUSAL_MESSAGE = "I can only answer questions based on Zyro Dynamics HR policy documents."

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are the official Zyro Dynamics HR Assistant. Provide highly accurate, professional, and concise answers based strictly on the provided context.

Context:
{context}

Question:
{question}

Strict Rules:
1. Base your answer ONLY on the provided context.
2. Extract exact numbers, timelines, eligibility tiers, grades, and specific criteria.
3. If the context does not explicitly contain the answer, reply exactly with:
I can only answer questions based on Zyro Dynamics HR policy documents.

Answer:
""")

def format_docs(docs):
    return "\n\n".join(
        f"[Source: {os.path.basename(doc.metadata.get('source', 'Policy Document'))}]:\n{doc.page_content}"
        for doc in docs
    )

def process_user_query(question):
    docs = retriever.invoke(question)
    if not docs:
        return REFUSAL_MESSAGE, []

    context = format_docs(docs)

    eval_prompt = f"""
Determine if the provided context contains explicit factual info to accurately answer the employee's specific question.
Question: {question}
Context: {context}

Output exactly JSON format with a single key "viable" matching either true or false.
Example: {{"viable": true}} or {{"viable": false}}
Do not write prose.
"""

    try:
        eval_resp = llm.invoke(eval_prompt).content.strip()

        if "```json" in eval_resp:
            eval_resp = eval_resp.split("```json")[1].split("```")[0].strip()
        elif "```" in eval_resp:
            eval_resp = eval_resp.split("```")[1].split("```")[0].strip()

        if not json.loads(eval_resp).get("viable", False):
            return REFUSAL_MESSAGE, []

    except:
        if "false" in eval_resp.lower():
            return REFUSAL_MESSAGE, []

    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return answer, docs

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about leaves, travel reimbursements, work-from-home rules..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Scanning internal knowledge vectors..."):
            ans, source_docs = process_user_query(prompt)
            st.write(ans)

            if source_docs and ans != REFUSAL_MESSAGE:
                with st.expander("📚 Verified Policy References"):
                    for d in source_docs:
                        src_name = os.path.basename(d.metadata.get('source', 'Policy Doc'))
                        st.markdown(f"**Source Document:** `{src_name}`")
                        st.caption(d.page_content)
                        st.markdown("---")

    st.session_state.messages.append({"role": "assistant", "content": ans})
