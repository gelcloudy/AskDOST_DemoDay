import os
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_qdrant import Qdrant
from tavily import TavilyClient

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from datetime import datetime

# --- Load API keys from .env ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- LLM Setup ---
primary_qa_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# -----------------------------
# 1. CACHE HEAVY OBJECTS
# -----------------------------
@st.cache_resource
def load_llm():
    return ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

@st.cache_resource
def load_vectorstore():
    pdf_files = [
        "data/2025-DOST-SEI-ST-Scholars-Handbook-Final.pdf",
        #"data/2024-ASTHRDP-Brochure.pdf",
        #"data/2024cbpsmeBroc.pdf",
        #"data/2024cbpsmePart.pdf",
        #"data/2024erdtBroc.pdf",
        #"data/2024StrandBrochure.pdf",
    ]

    all_documents = []
    for pdf in pdf_files:
        loader = PyMuPDFLoader(pdf)
        docs = loader.load()
        all_documents.extend(docs)

    # --- Step 2: Split documents into chunks ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(all_documents)

    # --- Step 3: Create embeddings ---
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # --- Step 4: Build in-memory vectorstore ---
    vectorstore = Qdrant.from_documents(
        chunks,
        embeddings,
        location=":memory:",
        collection_name="DOST_Handbook",
    )

    return vectorstore

# --- Init core retrievers ---
llm = load_llm()
qdrant_vector_store = load_vectorstore()

retriever = qdrant_vector_store.as_retriever(search_kwargs={"k": 7})
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever,
)
advanced_retriever = MultiQueryRetriever.from_llm(
    retriever=compression_retriever,
    llm=llm,
)

# --- Custom Prompt (encourages detailed answers) ---
from langchain.prompts import ChatPromptTemplate

handbook_prompt = PromptTemplate.from_template( """
You are an assistant for DOST-SEI scholars ans wering questions using the official handbook.

CRITICAL INSTRUCTIONS:
1. Answer ONLY based on the provided context below
2. Do NOT add information not present in the context
3. If you're unsure, say "I need more specific information from the handbook" or "Can you give mo more context to your question?"
4. Quote specific sections when possible

Context:
{context}

Question:
{input}

Answer clearly and concisely.
""" )

handbook_chain = LLMChain(llm=primary_qa_llm, prompt=handbook_prompt)


# --- Document Chain and Retrieval Chain ---

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import LLMChain, create_retrieval_chain

document_chain = create_stuff_documents_chain(primary_qa_llm, handbook_prompt)
retrieval_chain = create_retrieval_chain(advanced_retriever, document_chain)

# --- Guardrails ---
def guardrails_pre_query(user_input: str) -> str:
    blocked = ["violence", "politics", "religion"]
    if any(b in user_input.lower() for b in blocked):
        return "Sorry, I cannot answer that type of question."
    return user_input

def guardrails_post_response(response: str) -> str:
    if "I don't know" in response or len(response.strip()) < 20:
        response += "\n\n⚠️ This answer may be incomplete. Please verify in the official handbook."
    return response


# --- Judge Chain ---
fallback_prompt = PromptTemplate.from_template("""
You are a judge
Given the user question and the handbook answer,
decide if the answer is sufficient, accurate, and complete.

Question: {question}
Answer: {answer}

Respond with ONLY one word: "sufficient" or "insufficient".
""")
judge_chain = LLMChain(llm=primary_qa_llm, prompt=fallback_prompt)


from datetime import datetime
from tavily import TavilyClient

tavily = TavilyClient(api_key=TAVILY_API_KEY)

def tavily_search(query: str, max_results: int = 5) -> tuple[str, list]:
    # Step 1: Try trusted domains
    preferred_domains = [
        "sei.dost.gov.ph",
        "dost.gov.ph",
        "facebook.com/DOST.SEI",
        "facebook.com/DOSTph",
        "science-scholarships.ph",
    ]

    results = tavily.search(
        query,
        max_results=max_results,
        include_domains=preferred_domains
    )

    # Step 2: Fall back to open web if nothing
    if not results["results"]:
         results = tavily.search(query, max_results=max_results)

    # Step 3: Format with dates
    parsed = []
    summaries = []
    for r in results["results"]:
        pub_date = r.get("published_date")
        if pub_date:
            try:
                pub_date = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                date_str = pub_date.strftime("%Y-%m-%d")
            except Exception:
                date_str = "N/A"
        else:
            date_str = "N/A"

        summaries.append(f"[{date_str}] {r['title']}: {r['content']}")
        parsed.append({"title": r["title"], "url": r["url"], "date": date_str})

    return "\n".join(summaries), parsed


# --- Stronger web prompt that requires recency & citations ---
web_prompt = PromptTemplate.from_template("""
You are an assistant for DOST-SEI scholars. Use ONLY the top search results below to answer the question.
IMPORTANT:
- PRIORITIZE the most recent information.
- If the results do not contain recent/definitive information, reply exactly: "I couldn't find recent information on this topic."

Top search results (most recent first):
{results}

Question:
{question}

Answer (use only the search results above):
""")

web_chain = LLMChain(llm=primary_qa_llm, prompt=web_prompt)

@st.cache_data(show_spinner="Answering ...")
def hybrid_agent(question: str, history: list = None) -> str:
    """
    Hybrid agent with handbook RAG + web fallback.
    Accepts optional conversation history for follow-ups.
    Handles Tavily query length limits.
    """

    # --- Step 0: Use conversation memory ---
    context_from_history = ""
    if history:
        # Take only last 3 exchanges to keep context focused
        context_from_history = "\n\nRecent conversation:\n" + "\n".join(
            [f"{m['role']}: {m['content']}" for m in history[-3:]]
        )

    # Append history to user question
    full_query = question + context_from_history

    # --- Step 1: Guardrails (pre-query) ---
    safe_query = guardrails_pre_query(full_query)
    if safe_query.startswith("Sorry"):
        return safe_query

    # --- Step 2: Handbook RAG ---
    handbook_response = retrieval_chain.invoke({"input": safe_query})
    answer = handbook_response["answer"]
    answer = guardrails_post_response(answer)

    # --- Step 3: Judge sufficiency ---
    judge_result = judge_chain.invoke({"question": safe_query, "answer": answer})
    if "insufficient" in judge_result["text"].strip().lower():
        # Prepare a shorter query for Tavily to avoid 400-character limit
        tavily_query = question
        if history:
            # Keep last 1–2 user messages only
            recent_user_msgs = [m['content'] for m in history if m['role'] == "user"][-2:]
            tavily_query += " " + " ".join(recent_user_msgs)
        tavily_query = tavily_query[:300]  # truncate to 300 chars

        results_text, parsed_results = tavily_search(tavily_query)
        web_answer = web_chain.invoke({
            "results": results_text,
            "question": tavily_query
        })
        return web_answer["text"]

    return answer
