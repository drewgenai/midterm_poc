import os
import shutil
import json
import pandas as pd
import chainlit as cl
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from langchain.tools import tool
from langchain.schema import HumanMessage
from typing_extensions import List, TypedDict
from operator import itemgetter

# Load environment variables
load_dotenv()

# Define paths
UPLOAD_PATH = "upload/"
OUTPUT_PATH = "output/"
os.makedirs(UPLOAD_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Initialize embeddings model
model_id = "Snowflake/snowflake-arctic-embed-m"
embedding_model = HuggingFaceEmbeddings(model_name=model_id)

# Define semantic chunker
semantic_splitter = SemanticChunker(embedding_model)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Define RAG prompt
export_prompt = """
CONTEXT:
{context}

QUERY:
{question}

You are a helpful assistant. Use the available context to answer the question.

Between these two files containing protocols, identify and match **entire assessment sections** based on conceptual similarity. Do NOT match individual questions.

### **Output Format:**
Return the response in **valid JSON format** structured as a list of dictionaries, where each dictionary contains:
[
    {{
        "Derived Description": "A short name for the matched concept",
        "Protocol_1": "Protocol 1 - Matching Element",
        "Protocol_2": "Protocol 2 - Matching Element"
    }},
    ...
]
### **Example Output:**
[
    {{
        "Derived Description": "Pain Coping Strategies",
        "Protocol_1": "Pain Coping Strategy Scale (PCSS-9)",
        "Protocol_2": "Chronic Pain Adjustment Index (CPAI-10)"
    }},
    {{
        "Derived Description": "Work Stress and Fatigue",
        "Protocol_1": "Work-Related Stress Scale (WRSS-8)",
        "Protocol_2": "Occupational Fatigue Index (OFI-7)"
    }},
    ...
]

### Rules:
1. Only output **valid JSON** with no explanations, summaries, or markdown formatting.
2. Ensure each entry in the JSON list represents a single matched data element from the two protocols.
3. If no matching element is found in a protocol, leave it empty ("").
4. **Do NOT include headers, explanations, or additional formatting**—only return the raw JSON list.
5. It should include all the elements in the two protocols.
6. If it cannot match the element, create the row and include the protocol it did find and put "could not match" in the other protocol column.
7. protocol should be the between
"""

compare_export_prompt = ChatPromptTemplate.from_template(export_prompt)

QUERY_PROMPT = """
You are a helpful assistant. Use the available context to answer the question concisely and informatively.

CONTEXT:
{context}

QUERY:
{question}

Provide a natural-language response using the given information. If you do not know the answer, say so.
"""

query_prompt = ChatPromptTemplate.from_template(QUERY_PROMPT)


@tool
def document_query_tool(question: str) -> str:
    """Retrieves relevant document sections and answers questions based on the uploaded documents."""

    retriever = cl.user_session.get("qdrant_retriever")
    if not retriever:
        return "Error: No documents available for retrieval. Please upload documents first."

    # Retrieve context from the vector database
    retrieved_docs = retriever.invoke(question)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Generate response using the natural query prompt
    messages = query_prompt.format_messages(question=question, context=docs_content)
    response = llm.invoke(messages)

    return {
        "messages": [HumanMessage(content=response.content)],
        "context": retrieved_docs
    }



@tool
def document_comparison_tool(question: str) -> str:
    """Compares documents, identifies matched elements, exports them as JSON, formats into CSV, and provides a download link."""

    # Retrieve the vector database retriever
    retriever = cl.user_session.get("qdrant_retriever")
    if not retriever:
        return "Error: No documents available for retrieval. Please upload two PDF files first."

    # Process query using RAG
    rag_chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | compare_export_prompt | llm | StrOutputParser()
    )
    response_text = rag_chain.invoke({"question": question})

    # Parse response and save as CSV
    try:
        structured_data = json.loads(response_text)
        if not structured_data:
            return "Error: No matched elements found."

        # Define output file path
        file_path = os.path.join(OUTPUT_PATH, "comparison_results.csv")

        # Save to CSV
        df = pd.DataFrame(structured_data, columns=["Derived Description", "Protocol_1", "Protocol_2"])
        df.to_csv(file_path, index=False)

        return file_path  # Return path to the CSV file

    except json.JSONDecodeError:
        return "Error: Response is not valid JSON."



tool_belt = [document_query_tool, document_comparison_tool]
model = ChatOpenAI(model="gpt-4o", temperature=0)
model = model.bind_tools(tool_belt)

async def process_files(files: list[cl.File]):
    documents_with_metadata = []
    for file in files:
        file_path = os.path.join(UPLOAD_PATH, file.name)
        shutil.copyfile(file.path, file_path)
        
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        for doc in documents:
            source_name = file.name
            chunks = semantic_splitter.split_text(doc.page_content)
            for chunk in chunks:
                doc_chunk = Document(page_content=chunk, metadata={"source": source_name})
                documents_with_metadata.append(doc_chunk)
    
    if documents_with_metadata:
        qdrant_vectorstore = Qdrant.from_documents(
            documents_with_metadata,
            embedding_model,
            location=":memory:",
            collection_name="document_comparison",
        )
        return qdrant_vectorstore.as_retriever()
    return None

@cl.on_chat_start
async def start():
    cl.user_session.set("qdrant_retriever", None)
    files = await cl.AskFileMessage(
        content="Please upload **two PDF files** for comparison:",
        accept=["application/pdf"],
        max_files=2
    ).send()
    
    if len(files) != 2:
        await cl.Message("Error: You must upload exactly two PDF files.").send()
        return
    
    retriever = await process_files(files)
    if retriever:
        cl.user_session.set("qdrant_retriever", retriever)
        await cl.Message("Files uploaded and processed successfully! You can now enter your query.").send()
    else:
        await cl.Message("Error: Unable to process files. Please try again.").send()

@cl.on_message
async def handle_message(message: cl.Message):
    user_input = message.content.lower()

    # If the user asks for a comparison, run the document_comparison_tool
    if "compare" in user_input or "export" in user_input:
        file_path = document_comparison_tool.invoke(user_input)
        if file_path and file_path.endswith(".csv"):
            await cl.Message(
                content="Comparison complete! Download the CSV below:",
                elements=[cl.File(name="comparison_results.csv", path=file_path, display="inline")],
            ).send()
        else:
            await cl.Message(file_path).send()
    else:
        response_text = document_query_tool.invoke(user_input)
        await cl.Message(response_text["messages"][0].content).send()
