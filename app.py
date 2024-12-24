import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI

os.environ['GOOGLE_API_KEY'] = {Enter API key here}


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


def get_conversational_chain(vector_store):
    llm = GoogleGenerativeAI(model="models/text-bison-001", temperature=0.7, max_tokens=2000)  # Increase max_tokens
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain



def user_input(user_question):
    if user_question:
        response = st.session_state.conversation({'question': user_question})
        # Check if response is valid and contains an answer
        if 'answer' in response and response['answer']:
            # Add the new question and answer to chat history
            st.session_state.chatHistory.append({
                'question': user_question,
                'answer': response['answer']
            })
            # Display the updated chat history
            left, right = st.columns((2, 1))
            with left:
                for chat in st.session_state.chatHistory:
                    st.write("Human:", chat['question'])
                    st.write("Bot:", chat['answer'])
            with right:
                file_contents = "\n".join(
                    f"Human: {chat['question']}\nBot: {chat['answer']}"
                    for chat in st.session_state.chatHistory
                )
                file_name = "Chat_History.txt"
                st.download_button("Download chat history👈", file_contents, file_name=file_name, mime="text/plain")
        else:
            st.error("No answer was generated. Please try a different question.")

def main():
    st.set_page_config("Chat with Multiple PDFs")
    st.header("Chat with Multiple PDFs 💬")
    st.write("---")
    with st.container():
        with st.sidebar:
            st.title("Settings")
            st.subheader("Upload your Documents")
            pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Process Button", accept_multiple_files=True)
            if st.button("Process"):
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    st.session_state.chatHistory = []  # Initialize or clear chat history
                    st.success("Done")
    with st.container():
        # Question Section
        st.subheader("PDF question-answer section")
        user_question = st.text_input("Ask a Question from the PDF Files")
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if st.button("Generate Answer"):
            user_input(user_question)

if __name__ == "__main__":
    main()
