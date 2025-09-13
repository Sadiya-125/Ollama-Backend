import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage

# ------------------------
# Config
# ------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medical-chatbot"

# ------------------------
# Init Services
# ------------------------
app = FastAPI()

# LLM + Embedding
llm = Ollama(model="mistral")
embedding = OllamaEmbeddings(model="all-minilm")

# Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Load vector store
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embedding
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ------------------------
# Prompt Templates
# ------------------------
contextualize_q_system_prompt = """Given a Chat History and the Latest User Question 
Which Might Reference Context in the Chat History, Formulate a Standalone Question 
Which can be Understood without the Chat History. Do NOT Answer the Question, 
just Reformulate it if Needed and Otherwise Return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

qa_system_prompt = """You are a Medical Assistant for Question-Answering Tasks. 
Use the Following Pieces of Retrieved Context to Answer the Question. 
If you Don't Know the Answer, Just Say that you Don't Know. 
Use Three Sentences Maximum and Keep the Answer Concise.

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# ------------------------
# Helper Functions
# ------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain.invoke(
            {"chat_history": input["chat_history"], "question": input["question"]}
        )
    return input["question"]

rag_chain = (
    RunnablePassthrough.assign(
        context=lambda input: format_docs(
            retriever.get_relevant_documents(contextualized_question(input))
        )
    )
    | qa_prompt
    | llm
)

# ------------------------
# API Models
# ------------------------
class ChatRequest(BaseModel):
    question: str
    chat_history: list = []  # Pass prior conversation

class ChatResponse(BaseModel):
    answer: str

# ------------------------
# API Endpoint
# ------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    response = rag_chain.invoke(
        {"question": req.question, "chat_history": req.chat_history}
    )
    return ChatResponse(answer=response)