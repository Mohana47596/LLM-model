from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

# Define the path to the FAISS database and the Llama model
DB_FAISS_PATH = r"vectorstore/db_faiss"
LLAMA_MODEL_PATH = r"C:\Users\mohan\Videos\NEW VOLUME E\nydhackathon\llama-2-7b-chat.ggmlv3.q8_0.bin"

# Custom prompt template for QA
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type='stuff',
                                            retriever=db.as_retriever(search_kwargs={'k': 2}),
                                            return_source_documents=True,
                                            chain_type_kwargs={'prompt': prompt})
    return qa_chain

# Initialize the LLM model
llm = CTransformers(
    model=r"C:\Users\mohan\Videos\NEW VOLUME E\nydhackathon\llama-2-7b-chat.ggmlv3.q8_0.bin",
    model_type="llama",
    max_new_tokens=256,  # Reduced tokens
    temperature=0.7
)

# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)  # Allow dangerous deserialization

    # Reuse the Llama model
    qa_prompt = set_custom_prompt()
    print("SAMAY bot initialized successfully")
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

# Chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting SAMAY bot...")
    await msg.send()
    msg.content = "Hi, Welcome to SAMAY - Spiritual Assistance and Meditation Aid for You. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    if chain is None:
        await message.reply("Error: Chain not initialized properly. Please try again later.")
        return

    res = await chain.ainvoke(message.content)

    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()
