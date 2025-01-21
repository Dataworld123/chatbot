import os
from flask import Flask, render_template, request
from dotenv import load_dotenv

# LangChain imports
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Flask app initialization
app = Flask(__name__)

# Load the Gemini API key from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment.")

# Initialize the LLM (ChatGoogleGenerativeAI)
llm = ChatGoogleGenerativeAI(
    api_key=GEMINI_API_KEY,
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Load documents for retrieval
loader = WebBaseLoader(["https://github.com/Dataworld123/manodata/blob/main/requirements.txt/"])
loader.requests_per_second = 1
docs = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Assign unique IDs to splits
for i, doc in enumerate(splits):
    if "id" not in doc.metadata:
        doc.metadata["id"] = f"doc_{i}"

# Initialize embeddings and retriever
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# History-aware retriever configuration
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# QA Chain
system_prompt = (
    "You are a Manoindia support assistant for question-answering tasks. "
    "When asked 'Who are you?', reply 'I am Manoindia support assistant.' "
    "Use the following retrieved context to answer the question. If you don't know the answer, say so. "
    "Answer concisely in three sentences."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Retrieval Chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Session management
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Combine RAG chain with session management
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Flask Routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    input_data = {"input": msg}

    # Use a static session_id for simplicity; replace with unique user session handling
    session_id = "abc123"

    try:
        response = conversational_rag_chain.invoke(
            input_data,
            config={"configurable": {"session_id": session_id}}
        )
        answer = response.get("answer", "No response available.")
    except Exception as e:
        answer = f"Error: {str(e)}"

    return str(answer)
if __name__ == '__main__':
    app.run(debug=True)

