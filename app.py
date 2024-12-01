import os
import streamlit as st
import pickle
import json
import nltk
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import faiss 
from transformers import pipeline, BartTokenizerFast
from bs4 import BeautifulSoup
import torch
import warnings
from langchain.prompts import PromptTemplate
nltk.download('punkt')
warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_default_tensor_type(torch.DoubleTensor)
load_dotenv()

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'conversation_context' not in st.session_state:
    st.session_state.conversation_context = []

st.title("Arch AI")

st.sidebar.info("Enter up to three URLs to process the content. Click 'Process URLs' to prepare the content for the chatbot. Use 'Summarize Articles' to generate brief summaries of the content.")

url1 = st.sidebar.text_input("Enter URL 1")
url2 = st.sidebar.text_input("Enter URL 2")
url3 = st.sidebar.text_input("Enter URL 3")

process_urls_clicked = st.sidebar.button("Process URLs")
summarize_articles_clicked = st.sidebar.button("Summarize Articles")

index_file_path = "faiss_index"
docstore_file = "docstore.pkl"
index_to_docstore_id_file = "index_to_docstore_id.pkl"
active_learning_file = "interactions_dataset.jsonl"
status_placeholder = st.empty()

llm = ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0.8,
    max_tokens=1000
)

def process_urls(urls):
    try:
        all_docs = []
        scraped_files = []
        if os.path.exists(docstore_file) and os.path.exists(index_to_docstore_id_file):
            with open(docstore_file, "rb") as f:
                docstore = pickle.load(f)
            existing_docs = list(docstore._dict.values())
            all_docs.extend(existing_docs)
        else:
            docstore = None

        for idx, url in enumerate(urls, start=1):
            if not url.strip():
                continue
            loader = UnstructuredURLLoader(urls=[url])
            data = loader.load()
            scraped_data_filename = f"scraped_data_{idx}.txt"
            with open(scraped_data_filename, "w", encoding="utf-8") as f:
                for doc in data:
                    f.write(doc.page_content + "\n\n")
            scraped_files.append(scraped_data_filename)
            all_docs.extend(data)
        if not all_docs:
            status_placeholder.error("No valid URLs provided for processing.")
            return
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        docs = text_splitter.split_documents(all_docs)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        faiss.write_index(vectorstore.index, index_file_path)
        with open(docstore_file, "wb") as f:
            pickle.dump(vectorstore.docstore, f)
        with open(index_to_docstore_id_file, "wb") as f:
            pickle.dump(vectorstore.index_to_docstore_id, f)
        st.session_state.processing_complete = True
        status_placeholder.success("URLs processed successfully!")
    except Exception as e:
        status_placeholder.error(f"Error processing URLs: {str(e)}")
        
# Summarization Model
@st.cache_resource
def load_summarizer():
    model_name = "sshleifer/distilbart-xsum-12-6"
    summarizer = pipeline(
        "summarization",
        model=model_name,
        tokenizer=model_name,
        device=0 if torch.cuda.is_available() else -1,
        batch_size=16,
        truncation=True
    )
    tokenizer = BartTokenizerFast.from_pretrained(model_name)
    return summarizer, tokenizer

def clean_text(text):
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = ' '.join(text.split())
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if len(line.split()) > 3 and not any(boilerplate in line.lower() for boilerplate in ['cookie', 'privacy policy', 'terms of service']):
            lines.append(line)
    return ' '.join(lines)
# Splitting the content into small chunks
def smart_chunking(text, tokenizer, max_tokens=1024):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence)
        sentence_length = len(sentence_tokens)
        if current_length + sentence_length > max_tokens:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                chunks.append(tokenizer.decode(sentence_tokens[:max_tokens], skip_special_tokens=True))
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def summarize_articles(urls):
    try:
        status_placeholder.info("Generating summaries...")
        valid_urls = [url for url in urls if url.strip()]
        if not valid_urls:
            status_placeholder.error("No URLs have been processed for summarization.")
            return
        summaries = {}
        summarizer, tokenizer = load_summarizer()
        for idx, url in enumerate(valid_urls, start=1):
            scraped_data_filename = f"scraped_data_{idx}.txt"
            if not os.path.exists(scraped_data_filename):
                continue
            with open(scraped_data_filename, "r", encoding="utf-8") as f:
                content = f.read()
            cleaned_text = clean_text(content)
            if not cleaned_text.strip():
                continue
            chunks = smart_chunking(cleaned_text, tokenizer, max_tokens=2048)
            summaries_list = []
            for chunk in chunks:
                summary = summarizer(
                    chunk,
                    max_length=400,
                    min_length=100,
                    do_sample=False
                )[0]['summary_text']
                summaries_list.append(summary)
            combined_summary = " ".join(summaries_list)
            final_summary = combined_summary
            summary_filename = f"summary_{idx}.txt"
            with open(summary_filename, "w", encoding="utf-8") as f:
                f.write(final_summary)
            summaries[url] = final_summary
        st.session_state.initial_message_displayed = False

        st.subheader("Summaries")
        for idx, url in enumerate(valid_urls, start=1):
            summary_filename = f"summary_{idx}.txt"
            if os.path.exists(summary_filename):
                with open(summary_filename, "r", encoding="utf-8") as f:
                    summary_text = f.read()
                st.write(f"**Summary for URL {idx}: {url}**")
                st.write(summary_text)
                st.markdown("---")
    except Exception as e:
        status_placeholder.error(f"Error generating summaries: {str(e)}")

if process_urls_clicked:
    urls = [url1, url2, url3]
    if not any(url.strip() for url in urls):
        status_placeholder.error("Please enter at least one valid URL.")
    else:
        st.session_state.initial_message_displayed = False
        status_placeholder.info("Processing URLs...")
        process_urls(urls)

if summarize_articles_clicked:
    urls = [url1, url2, url3]
    summarize_articles(urls)

if st.session_state.processing_complete:
    for chat in st.session_state.messages:
        if chat['role'] == 'user':
            with st.chat_message("user"):
                st.write(chat['content'])
        elif chat['role'] == 'assistant':
            with st.chat_message("assistant"):
                st.write(chat['content'])
                if 'sources' in chat and chat['sources']:
                    st.markdown(f"**Sources:** {chat['sources']}")
    user_input = st.chat_input("Type your message here...")
    if user_input:
        st.session_state.messages.append({'role': 'user', 'content': user_input})
        with st.chat_message("user"):
            st.write(user_input)
        try:
            index = faiss.read_index(index_file_path)
            with open(docstore_file, "rb") as f:
                docstore = pickle.load(f)
            with open(index_to_docstore_id_file, "rb") as f:
                index_to_docstore_id = pickle.load(f)
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS(embeddings.embed_query, index, docstore, index_to_docstore_id=index_to_docstore_id)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
            if os.path.exists(active_learning_file):
                with open(active_learning_file, "r", encoding="utf-8") as f:
                    few_shot_examples = [json.loads(line) for line in f.readlines()]
                custom_template = """
                You are an AI assistant trained to provide helpful and informative responses. Your goal is to assist users with their queries while being concise and clear. When answering, consider the following examples:

                {examples}

                Now, respond to the user's question:

                User: {{question}}

                Assistant:
                """
                formatted_examples = "\n\n".join([
                    f"User: {example['prompt']}\nAssistant: {example['completion']}"
                    for example in few_shot_examples[-5:]
                ])
                prompt_template = PromptTemplate.from_template(
                    custom_template.format(examples=formatted_examples)
                )
                result = chain({"question": user_input}, return_only_outputs=True)
            else:
                result = chain({"question": user_input}, return_only_outputs=True)
            answer = result.get("answer", "No answer found.")
            sources = result.get("sources", "")
            st.session_state.messages.append({'role': 'assistant', 'content': answer, 'sources': sources})
            with open(active_learning_file, "a", encoding="utf-8") as f:
                json_line = json.dumps({"prompt": user_input, "completion": answer})
                f.write(json_line + "\n")
            with st.chat_message("assistant"):
                st.write(answer)
                if sources:
                    st.markdown(f"**Sources:** {sources}")
                else:
                    st.markdown("**Sources:** No sources provided.")
        except Exception as e:
            status_placeholder.error(f"Error querying the model: {str(e)}")
