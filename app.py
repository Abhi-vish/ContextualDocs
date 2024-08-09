# Importing Liabraries
import streamlit as st
import pickle
import base64
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from main import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
from langchain.memory import ConversationBufferMemory
# from login import Login
from cryptography.fernet import Fernet
import firebase_admin
from firebase_admin import credentials, firestore


# Login class
class Login:
    def __init__(self):
        with st.sidebar:
            self.username = st.text_input("Username")
            self.password = st.text_input("Password", type="password")
            self.login_button = st.button("Login")
            self.register_button = st.button("Register")

    # Method to initialize connection to Firestore
    def initialize_connection(self):
        if not firebase_admin._apps:
            cred = credentials.Certificate("chatwithdoc-1de20-firebase-adminsdk-3cl78-70738bde08.json")
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        return db

    # Method to login user
    def login(self):
        db = self.initialize_connection()
        users_ref = db.collection("users")
        users = users_ref.stream()

        for user in users:
            user_data = user.to_dict()
            if user_data['username'] == self.username and user_data['password'] == self.password:
                st.session_state.user_id = user.id  # Save user ID in session state
                return True
        return False

    # Method to register user
    def register(self):
        db = self.initialize_connection()
        users_ref = db.collection("users")
        if self.username and self.password:
            # Ensure username is unique
            existing_users = users_ref.where("username", "==", self.username).stream()
            if any(existing_users):
                st.error("Username already exists.")
                return

            # Add new user
            user_ref = users_ref.add({
                "username": self.username,
                "password": self.password
            })
            # st.session_state.user_id = user_ref.id  # Save user ID in session state
            st.success("Registration successful!")
        else:
            st.error("Please enter a valid username and password.")

    # Method to get current user ID
    def get_current_user_id(self):
        return st.session_state.get('user_id')

# Initialize Fernet encryption
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# Initialize login system
login_system = Login()

# Create a session state variable to store login status
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if login_system.login_button:
    if login_system.login():
        st.success("Login successful!")
        st.session_state.logged_in = True
    else:
        st.error("Login failed. Check your username and password.")

if st.session_state.logged_in:
    # Sidebar contents
    with st.sidebar:
        st.title('🤗💬 LLM Chat App')
        add_vertical_space(2)
        st.write('Chat History!')

        # Get current user ID
        user_id = login_system.get_current_user_id()
        db = login_system.initialize_connection()
        doc_ref = db.collection('users').document(user_id)

        
        # Fetch conversation documents from Firestore
        conversation = doc_ref.collection('conversations').stream()
        question = []

        # Loop through the conversations and collect question IDs
        for convo in conversation:
            question.append(convo.id)
        
        # Display the selectbox with the collected question IDs
        selected_question = st.selectbox("Select a question", question)

        # If a question is selected, fetch and display the conversation
        if selected_question:
            conversation_ref = doc_ref.collection('conversations').document(selected_question)
            conversation = conversation_ref.get().to_dict()

            if conversation:
                st.write(f"Question: {conversation['question']}")
                st.write(f"Response: {conversation['response']}")

# Main function
def main():
    st.header("Chat with PDF 💬")

    # upload a PDF file
    doc = st.file_uploader("Upload your PDF", type=["pdf", "docx", "txt"])

    # If a document is uploaded
    if doc is not None:
        doc_format = doc.name.split('.')[-1]

        # Extract text from the document if it is pdf
        if doc_format == 'pdf':
            text = extract_text_from_pdf(doc)
        
        # Extract text from the document if it is docx
        elif doc_format == 'docx':
            text = extract_text_from_docx(doc)

        # Extract text from the document if it is txt
        elif doc_format == 'txt':
            text = extract_text_from_txt(doc)

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Encrypt and encode chunks to be JSON serializable
        encrypted_chunks = [base64.b64encode(cipher_suite.encrypt(chunk.encode())).decode('utf-8') for chunk in chunks]

        # Get current user's document reference
        user_id = login_system.get_current_user_id()
        docu_ref = db.collection('documents')

        docu_ref.add({'chunks': encrypted_chunks})

        store_name = doc.name[:-4]
        st.write(f'{store_name}')

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            # Create a vector store from the text chunks
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Ask questions about the PDF file
        query = st.text_input("Ask questions about your PDF file:")

        # If a query is entered
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            # Initialize the ChatOllama with the conversation buffer memory
            llm = ChatOllama(
                model="qwen2:1.5b",
                temperature=0,
                verbose=True
            )

            # Load the QA chain with the ChatOllama instance
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            # Run the QA chain with the input documents and query
            response = chain.run(input_documents=docs, question=query)

            # Display the response
            st.write(response)


            # Store conversation in the current user's collection
            doc_ref.collection('conversations').add(
                {
                    'question': query,
                    'response': response
                }
            )
            
# Run the main function
if __name__ == '__main__':
    main()

# Register user
if login_system.register_button:
    login_system.register()
