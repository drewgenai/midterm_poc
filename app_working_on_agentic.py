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
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import MessagesPlaceholder

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
        "answer": response.content,
        "context": retrieved_docs
    }


@tool
def document_comparison_tool(question: str) -> str:
    """Compares the two uploaded documents, identifies matched elements, exports them as JSON, formats into CSV, and provides a download link."""

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

        # Store the file path in the user session for later retrieval
        cl.user_session.set("comparison_file_path", file_path)
        
        return "Comparison complete! CSV file has been generated."

    except json.JSONDecodeError:
        return "Error: Response is not valid JSON."


# Define tools for the agent
tools = [document_query_tool, document_comparison_tool]

# Set up the agent with a system prompt
system_prompt = """You are an intelligent document analysis assistant. You have access to two tools:

1. document_query_tool: Use this when a user wants information or has questions about the content of uploaded documents.
2. document_comparison_tool: Use this when a user wants to compare elements between two uploaded documents or export comparison results.

Analyze the user's request carefully to determine which tool is most appropriate.
"""

# Create the agent using OpenAI function calling
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(
    llm=ChatOpenAI(model="gpt-4o", temperature=0),
    tools=tools,
    prompt=agent_prompt
)

# Create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)


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
    # Initialize chat history for the agent
    cl.user_session.set("chat_history", [])
    cl.user_session.set("qdrant_retriever", None)
    
    files = await cl.AskFileMessage(
        content="Please upload **two PDF files** for comparison:",
        accept=["application/pdf"],
        max_files=2
    ).send()
    
    if len(files) != 2:
        await cl.Message("Error: You must upload exactly two PDF files.").send()
        return
    
    with cl.Step("Processing files"):
        retriever = await process_files(files)
        if retriever:
            cl.user_session.set("qdrant_retriever", retriever)
            await cl.Message("Files uploaded and processed successfully! You can now enter your query.").send()
        else:
            await cl.Message("Error: Unable to process files. Please try again.").send()


@cl.on_message
async def handle_message(message: cl.Message):
    # Get chat history
    chat_history = cl.user_session.get("chat_history", [])
    
    # Run the agent
    with cl.Step("Agent thinking"):
        response = await cl.make_async(agent_executor.invoke)(
            {"input": message.content, "chat_history": chat_history}
        )
    
    # Handle the response based on the tool that was called
    output = response["output"]
    
    if isinstance(output, dict) and "answer" in output:
        # This is from document_query_tool
        await cl.Message(output["answer"]).send()
        
    elif "Comparison complete!" in str(output):
        # This is from document_comparison_tool
        file_path = cl.user_session.get("comparison_file_path")
        
        if file_path and os.path.exists(file_path):
            # Read the file content
            with open(file_path, "rb") as f:
                file_content = f.read()
            
            # Create a File element with the content
            file_element = cl.File(
                name="comparison_results.csv",
                content=file_content,
                display="inline"
            )
            
            # Send the message with the file element
            await cl.Message(
                content="Comparison complete! Download the CSV below:",
                elements=[file_element],
            ).send()
        else:
            await cl.Message(content=str(output)).send()
        
    else:
        # Generic response
        await cl.Message(content=str(output)).send()
    
    # Update chat history with the new exchange
    chat_history.extend([
        HumanMessage(content=message.content),
        HumanMessage(content=str(output))
    ])
    cl.user_session.set("chat_history", chat_history)