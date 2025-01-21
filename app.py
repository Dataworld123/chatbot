import os
from dotenv import load_dotenv
load_dotenv()


import os
os.environ['USER_AGENT'] = 'myagent'
from langchain_google_genai import ChatGoogleGenerativeAI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY,
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
   
)

"""messages = [
    (
        "system",
        "You are a helpful assistant that you give the answer in 20 word.",
    ),
    ("human", "who is virat kohli"),
]
ai_msg = llm.invoke(messages)

print(ai_msg.content)"""


import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

"""vector = embeddings.embed_query("hello, world!")
vector[:5]"""





"""loader = WebBaseLoader(
    web_paths=("https://github.com/Dataworld123/manodata/blob/main/requirements.txt/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)"""
loader = WebBaseLoader(["https://github.com/Dataworld123/manodata/blob/main/requirements.txt/"])
loader.requests_per_second = 1
docs = loader.aload()
print(docs)

docs = loader.load()

print(f"Loaded {len(docs)} documents")  



"""text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f"Generated {len(splits)} document splits") 

for i, doc in enumerate(splits):
    if "id" not in doc.metadata:
        doc.metadata["id"] = f"doc_{i}"  

vectorstore = Chroma.from_documents(documents=splits, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))


retriever = vectorstore.as_retriever()





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



system_prompt = (
    "You are Manoindia support  assistant for question-answering tasks. "
    "question who are you ,you give answer - i am Manoindia support assistant"
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
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

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)



store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

"""conversational_rag_chain.invoke(
    {"input": "What is Task Decomposition?"},
    config={
        "configurable": {"session_id": "abc123"}
    },  # constructs a key "abc123" in `store`.
)["answer"]

#print(reponse)"""





from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os


app = Flask(__name__)

load_dotenv()



@app.route("/")
def index():
    return render_template('chat.html')

"""@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    result=conversational_rag_chain.invoke(input)
    print("Response : ", result)
    return str(result)

if __name__ == '__main__':
    app.run(debug= True)"""

"""@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg

    # Provide a session_id in the configuration for each request
    session_id = "abc123"  # This should be unique per session/user ideally
    result = conversational_rag_chain.invoke(
        {"input": input},
        config={
            "configurable": {"session_id": session_id}
        }
    )
    ['answer']
    print("Response : ", result)
    return str(result)"""



if __name__ == '__main__':
    app.run(debug= True)

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input_data = {"input": msg}  # Corrected to use a dictionary as input

    # Provide a session_id in the configuration for each request
    session_id = "abc123"  # This should be unique per session/user ideally

    # Invoke the conversational RAG chain with both input and configuration
    response = conversational_rag_chain.invoke(
        input_data,
        config={
            "configurable": {"session_id": session_id}
        }
    )

 
    answer = response.get("answer", "")  # Safely get the 'answer' or default to empty

    print("Response : ", answer)  # For debugging purposes
    return str(answer)  # Return just the answer