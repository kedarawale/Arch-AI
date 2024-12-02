import os
import streamlit as st
import pickle
import json
import nltk
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import faiss 
from bs4 import BeautifulSoup
import torch
import warnings
from langchain.schema import Document  # Import Document class

# Download necessary NLTK data
nltk.download('punkt')

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set default tensor type for PyTorch
torch.set_default_tensor_type(torch.DoubleTensor)

# Load environment variables from .env file
load_dotenv()

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'conversation_context' not in st.session_state:
    st.session_state.conversation_context = []
if 'summaries' not in st.session_state:
    st.session_state.summaries = {}
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Set the title of the Streamlit app
st.title("Arch AI")

# Sidebar instructions and inputs
st.sidebar.info(
    "Enter up to three URLs to process the content. Click 'Process URLs' to prepare the content for the chatbot. "
    "Use 'Summarize Articles' to generate brief summaries of the content."
)

# Sidebar inputs for URLs
url1 = st.sidebar.text_input("Enter URL 1")
url2 = st.sidebar.text_input("Enter URL 2")
url3 = st.sidebar.text_input("Enter URL 3")

# Sidebar buttons
process_urls_clicked = st.sidebar.button("Process URLs")
summarize_articles_clicked = st.sidebar.button("Summarize Articles")

# File paths for FAISS index and docstore
index_file_path = "faiss_index"
docstore_file = "docstore.pkl"
index_to_docstore_id_file = "index_to_docstore_id.pkl"
active_learning_file = "interactions_dataset.jsonl"

# Placeholder for status messages
status_placeholder = st.empty()

# Initialize Language Model for Chat
llm = ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0.2,  # Lowered temperature for more deterministic responses
    max_tokens=1000
)

def process_urls(urls):
    """
    Processes the provided URLs by scraping their content, cleaning the text, splitting into chunks,
    and creating a new FAISS vector store. It resets any existing vector store to ensure only the current
    URLs are considered in the Q&A system.
    """
    try:
        all_docs = []
        scraped_files = []
        
        # **Reset Existing Data**
        # Remove existing FAISS index and docstore files to ensure only current URLs are processed
        if os.path.exists(index_file_path):
            os.remove(index_file_path)
        if os.path.exists(docstore_file):
            os.remove(docstore_file)
        if os.path.exists(index_to_docstore_id_file):
            os.remove(index_to_docstore_id_file)
        
        # Remove previously scraped data files to prevent mixing old and new content
        for i in range(1, 4):
            scraped_data_filename = f"scraped_data_{i}.txt"
            if os.path.exists(scraped_data_filename):
                os.remove(scraped_data_filename)
        
        # Scrape each URL and save content
        for idx, url in enumerate(urls, start=1):
            if not url.strip():
                continue
            loader = UnstructuredURLLoader(urls=[url])
            data = loader.load()
            if not data:
                status_placeholder.warning(f"No content found at URL {url}.")
                continue
                
            # Explicitly add source metadata to each document
            for doc in data:
                doc.metadata = {"source": url}  # Ensure source URL is properly set
            
            scraped_data_filename = f"scraped_data_{idx}.txt"
            with open(scraped_data_filename, "w", encoding="utf-8") as f:
                for doc in data:
                    f.write(doc.page_content + "\n\n")
            scraped_files.append(scraped_data_filename)
            all_docs.extend(data)
        
        if not all_docs:
            status_placeholder.error("No valid URLs provided for processing.")
            return
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1500,    # Increased chunk size to provide more context
            chunk_overlap=200    # Added overlap to maintain context between chunks
        )
        docs = text_splitter.split_documents(all_docs)
        
        # Create embeddings and vectorstore
        # Create a new FAISS vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_kwargs={
                    "k": 3,
                    "include_metadata": True
                }
            ),
            return_source_documents=True,  # Make sure to return source documents
            verbose=True
        )
        st.session_state.qa_chain = qa_chain
        
        # Reset chat messages
        st.session_state.messages = []
        
        # Clear previous summaries
        st.session_state.summaries = {}
        
        # Update session state
        
        st.session_state.processing_complete = True
        status_placeholder.success("URLs processed successfully!")
    except Exception as e:
        status_placeholder.error(f"Error processing URLs: {str(e)}")

# Summarization Function
@st.cache_resource
def load_summarizer():
    """
    Loads a simple summarization chain using LangChain.
    """
    # Retained BART model code (not used)
    # model_name_bart = "sshleifer/distilbart-xsum-12-6"
    # summarizer_bart = pipeline(
    #     "summarization",
    #     model=model_name_bart,
    #     tokenizer=model_name_bart,
    #     device=0 if torch.cuda.is_available() else -1,
    #     batch_size=16,
    #     truncation=True
    # )
    # tokenizer_bart = BartTokenizerFast.from_pretrained(model_name_bart)
    
    # Using LangChain's simple summarization chain
    summarize_chain = load_summarize_chain(llm, chain_type="refine")  # Changed chain_type to 'refine'
    return summarize_chain

def clean_text(text):
    """
    Cleans the input text by removing HTML tags and boilerplate content.
    """
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = ' '.join(text.split())
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if len(line.split()) > 3 and not any(
            boilerplate in line.lower() 
            for boilerplate in ['cookie', 'privacy policy', 'terms of service']
        ):
            lines.append(line)
    return ' '.join(lines)

def summarize_articles(urls):
    """
    Generates summaries for the processed URLs using the summarization chain.
    """
    try:
        status_placeholder.info("Generating summaries...")
        valid_urls = [url for url in urls if url.strip()]
        if not valid_urls:
            status_placeholder.error("No URLs have been processed for summarization.")
            return
        summaries = {}
        summarize_chain = load_summarizer()
        
        for idx, url in enumerate(valid_urls, start=1):
            scraped_data_filename = f"scraped_data_{idx}.txt"
            if not os.path.exists(scraped_data_filename):
                status_placeholder.warning(f"Scraped data file {scraped_data_filename} does not exist.")
                continue
            with open(scraped_data_filename, "r", encoding="utf-8") as f:
                content = f.read()
            cleaned_text = clean_text(content)
            if not cleaned_text.strip():
                status_placeholder.warning(f"No content to summarize for URL {idx}: {url}")
                continue
            # Wrap the cleaned text into a Document object with 'source' metadata
            documents = [Document(page_content=cleaned_text, metadata={"source": url})]
            # Run the summarization chain
            summary = summarize_chain.run(documents)
            summary_filename = f"summary_{idx}.txt"
            with open(summary_filename, "w", encoding="utf-8") as f:
                f.write(summary)
            summaries[url] = summary
        
        # Store summaries in session_state
        st.session_state.summaries = summaries
        
        # Update session state to indicate that summaries have been generated
        st.session_state.initial_message_displayed = False

        status_placeholder.success("Summaries generated successfully!")

    except Exception as e:
        status_placeholder.error(f"Error generating summaries: {str(e)}")

# Handle Process URLs button click
if process_urls_clicked:
    urls = [url1, url2, url3]
    if not any(url.strip() for url in urls):
        status_placeholder.error("Please enter at least one valid URL.")
    else:
        st.session_state.initial_message_displayed = False
        status_placeholder.info("Processing URLs...")
        process_urls(urls)

# Handle Summarize Articles button click
if summarize_articles_clicked:
    urls = [url1, url2, url3]
    summarize_articles(urls)

# Chat Interface
if st.session_state.processing_complete:
    # Display summaries at the top
    if st.session_state.summaries:
        st.subheader("Summaries")
        for idx, (url, summary) in enumerate(st.session_state.summaries.items(), start=1):
            st.write(f"**Summary for URL {idx}: {url}**")
            st.write(summary)
            st.markdown("---")
    
    # Display chat history below summaries
    for chat in st.session_state.messages:
        if chat['role'] == 'user':
            with st.chat_message("user"):
                st.write(chat['content'])
        elif chat['role'] == 'assistant':
            with st.chat_message("assistant"):
                st.write(chat['content'])
                if 'sources' in chat and chat['sources']:
                    st.markdown(f"**Sources:** {chat['sources']}")
    
    # User input for chat
    user_input = st.chat_input("Type your message here...")
    if user_input:
        st.session_state.messages.append({'role': 'user', 'content': user_input})
        with st.chat_message("user"):
            st.write(user_input)
        try:
            if st.session_state.qa_chain is None:
                status_placeholder.error("The QA system is not initialized. Please process URLs first.")
            else:
                # Run the QA chain with the user input
                result = st.session_state.qa_chain({"query": user_input})  # Note: using "query" instead of "question"
                
                # Extract the answer and source documents
                answer = result['result']  # Note: RetrievalQA uses 'result' instead of 'answer'
                source_documents = result['source_documents']
                
                # Get the source URL from the most relevant document
                sources_text = "No sources provided."
                
                if source_documents:
                    most_relevant_doc = source_documents[0]
                    if hasattr(most_relevant_doc, 'metadata') and 'source' in most_relevant_doc.metadata:
                        source_url = most_relevant_doc.metadata['source']
                        sources_text = f"Source: {source_url}"
                
                # Append assistant's response to chat history
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': answer,
                    'sources': sources_text
                })
                
                
                # Display assistant's response
                with st.chat_message("assistant"):
                    st.write(answer)
                    st.markdown(f"**{sources_text}**")
                
                # Log interaction for active learning
                with open(active_learning_file, "a", encoding="utf-8") as f:
                    json_line = json.dumps({
                        "prompt": user_input,
                        "completion": answer,
                    "sources": sources_text
                    })
                    f.write(json_line + "\n")
        except Exception as e:
            status_placeholder.error(f"Error querying the model: {str(e)}")
