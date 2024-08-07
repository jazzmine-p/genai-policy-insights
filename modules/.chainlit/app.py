import os
import yaml
from modules.data_loader import process_pdf
from modules.config.constants import chatbot_dir
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

from langchain.vectorstores.chroma import Chroma
from langchain.indexes import SQLRecordManager, index
from langchain.schema import Document
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig
from langchain.callbacks.base import BaseCallbackHandler

import chainlit as cl
import chainlit.data as cl_data

# Load configurations
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
embedding_config = config['embedding_model']['minilm-sm']
retriever_config = config['retriever']

chunk_size = 1024
chunk_overlap = 50

embedding_model = SentenceTransformer(embedding_config)

# Create embeddings and vector store
all_splits = process_pdf(chatbot_dir)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_model)

# Set up retriever with vector store
retriever = vectorstore.as_retriever(**retriever_config)
model = ChatOpenAI(model_name="gpt-4", streaming=True)

# Function to handle chat start
@cl.on_chat_start
async def on_chat_start():
    template = """You are an AI assistant for Generative AI Policy Insights, developed by the Boston University's GenAI Task Force. Your main mission is to help users understand how different organizations perceive and make policies regarding the use of GenAI. Answer the user's question using the provided context that is relevant. The context is ordered by relevance. 
                If you don't know the answer, do your best without making things up. If you cannot answer, just say you don't have enough relevant information to answer the questions. Keep the conversation flowing naturally. 
                Always cite the source of the information. Use the source context that is most relevant. 
                Keep the answer concise, yet professional and informative. Avoid sounding repetitive or robotic.\n
                Context:\n{context}\n\n
                Answer the user's question below in a friendly, concise, and engaging manner. Use the context and history only if relevant, otherwise, engage in a free-flowing conversation.\n
                User: {question}\n
                GenAI Policy Assistant:
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    runnable = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    cl.user_session.set("runnable", runnable)

# Function to handle incoming messages
@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")

    class PostMessageHandler(BaseCallbackHandler):
        """
        Callback handler for handling the retriever and LLM processes.
        Used to post the sources of the retrieved documents as a Chainlit element.
        """

        def __init__(self, msg: cl.Message):
            BaseCallbackHandler.__init__(self)
            self.msg = msg
            self.sources = set()  # To store unique pairs

        def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
            for d in documents:
                source_page_pair = (d.metadata['source'], d.metadata['page'])
                self.sources.add(source_page_pair)  

        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            if len(self.sources):
                sources_text = "\n".join([f"{source}#page={page}" for source, page in self.sources])
                self.msg.elements.append(
                    cl.Text(name="Sources", content=sources_text, display="inline")
                )

    async for chunk in runnable.astream(
        message.content,
        config=RunnableConfig(callbacks=[
            cl.LangchainCallbackHandler(),
            PostMessageHandler(msg)
        ]),
    ):
        await msg.stream_token(chunk)

    await msg.send()