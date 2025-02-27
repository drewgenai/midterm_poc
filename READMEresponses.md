## Task 1: Defining your Problem and Audience
### Problem Statement
Researchers at Studies Inc. must manually review multiple protocols, a time-consuming and complex process that requires scanning, extracting, and analyzing detailed information to identify patterns between studies.
### Why This is a Problem for Our Users
Researchers are spending an excessive amount of time manually reviewing study protocols to standardize data across multiple studies. Each protocol contains detailed and sometimes inconsistent information, making it difficult to quickly extract relevant data and compare it across different studies. The manual nature of this process increases the likelihood of human error, slows down research progress, and diverts time away from higher-value analytical work. Additionally, as the number of studies grows, this problem scales, further straining resources and delaying critical insights needed for scientific advancements.

## Task 2: Propose a Solution

The solution is to build an automated system that streamlines the process of reviewing and extracting key data from research protocols. By leveraging a combination of LLM-based retrieval-augmented generation (RAG) and structured orchestration, the system will allow researchers to quickly scan, extract, and compare relevant information across multiple protocols, significantly reducing manual effort and improving standardization.

Instead of using agents in the initial prototype, the solution will employ LangChain tools for structured execution, ensuring modular and predictable workflows. Chainlit will provide an event-driven UI that dynamically triggers relevant processing steps based on user interactions. This setup ensures efficiency, flexibility, and the ability to scale as more protocols and datasets are introduced.

While the initial implementation will rely on structured tool execution, future iterations or final builds may incorporate agentic reasoning to enhance adaptability. Agents could be employed for intelligent workflow management, dynamically selecting tools based on protocol complexity or user preferences. This could enable more sophisticated multi-step reasoning, where an agent determines the best extraction and comparison approach based on the type of research being conducted, but it is not required at this time with just two local documents.

We will build with the following stack:

- **LLM**: gpt-4o-mini  
We will leverage a closed-source model because the entire application is constructed using open-source data from public protocols
- **Embedding Models**: text-embedding-3-small, snowflake-arctic-embed-m  
We will construct a quick prototype using our closed-source OpenAI embedding model, and then we will fine-tune an off-the-shelf embedding model from Snowflake.  We want to demonstrate to leadership that it’s important to be able to work with open-source models and see that they can be as performant as closed-source while providing privacy benefits, especially after fine-tuning.*
- **Orchestration**: Langchain tools and tool based execution   
The application will orchestrate workflow execution using a structured, tools-based approach. The system will ensure predictable and efficient processing by defining modular tools (such as document comparison and retrieval) and explicitly invoking them in response to user input. Chainlit serves as the event-driven interface, managing tool execution based on user interactions, while LangChain tools handle document retrieval, RAG processing, and structured data export. This approach will provide flexibility and modularity.
- **Vector Store**: Qdrant  
It will be more than fast enough for our current data needs while maintaining low latency. Overall, it’s a solid, reliable choice, and we can take advantage of the fully open-source version at no cost and host it in memory running within our hugging face space.
- **Evaluation**: RAGAS  
RAGAS has been a leader in the AI evaluation space for years. We’re particularly interested in leveraging their RAG assessment metrics to test the performance of our embedding models.
- **User Interface**: Chainlit  
A lightweight, Python-based UIframework designed specifically for LLM applications. It allows us to quickly prototype and deploy conversational interfaces with minimal front-end effort while maintaining flexibility for customization.
- **Inference & Serving**: Hugging Face
We will leverage Hugging Face as a platform to serve up our application to users because it’s very fast (1-click deployment status) and very cheap to host.  Additionally, we will use Hugging Face to pull embedding models that we will test finetuning and host in the event it is needed.


## Task 3: Dealing with the Data
### Data Sources and External APIs
Our system will leverage multiple data sources and external APIs to extract, process, and analyze research protocols effectively. Below are the key sources and their roles in our application:

#### Uploaded Protocol Documents (PDFs)
- Researchers will upload research protocol documents in **PDF format**.
- These documents will be processed using **PyMuPDFLoader**, extracted as text, and chunked for embedding storage.
- We use **Qdrant** as the vector database to store document chunks for retrieval.

**Why?**  
Protocols contain structured and semi-structured data critical for comparison. By storing them in a vector database, we enable **semantic search and retrieval** for streamlined analysis.

---

#### Hugging Face API
- Used to access **Snowflake Arctic embedding models** for text embedding and potential fine-tuning.

**Why?**  
We aim to compare the performance of closed-source OpenAI embeddings with open-source models for **privacy, scalability, and long-term flexibility**.

---

#### OpenAI API (GPT-4o-mini)
- Used for generating **structured comparisons** between protocols.

**Why?**  
The LLM will process retrieved document chunks and generate **natural-language comparisons** and **structured JSON outputs**.

---

#### LangChain Tool Execution
- We use **LangChain’s tools-based execution** for structured retrieval and document comparison.

**Why?**  
Instead of an agentic approach, **explicit tool execution** ensures predictable and modular processing.  
Our priority at this time is **reducing API calls** while maintaining efficiency.

---

#### Chunking Strategy
- We will use a **semantic-based chunking strategy** with LangChain’s **SemanticChunker**, powered by **Snowflake Arctic embeddings**.

**Why this approach?**  
Traditional **fixed-size** or **recursive character-splitting** methods often break up conceptually linked sections.  
Semantic chunking ensures that **chunks maintain their conceptual integrity**, improving retrieval quality.

This method is especially useful for research protocols, where meaningful sections (e.g., **assessments, methodologies**) must remain intact for comparison.

As we refine the system, we may adjust chunk sizes based on **real user queries and retrieval performance** to optimize information density and response accuracy.

---

### Additional Data Needs
At present, our application focuses on **structured comparisons** between protocols. However, in future iterations, we may integrate additional data sources such as:

- **Metadata from Public Research Repositories** (e.g., PubMed, ArXiv API)  
  → To enrich protocol comparisons with **relevant external research**.
- **Institutional Databases** (if access is provided)  
  → To validate protocol **consistency across multi-site studies**.

While the current system is **not agentic**, we may explore **agent-based reasoning** in future versions to dynamically adjust retrieval and processing strategies based on protocol complexity.  At this time all the information we need is in the provided local documents.


### Agentic reasoning POC
There is a separate poc in the app_working_on_agentic.py file.  That utilizes the agentic approach. 
It has an agent executor that utilizes agentic reasoning in this application by leveraging OpenAI's function calling to dynamically choose and execute the most appropriate tool based on the user's input. The agent is guided by a predefined system prompt that outlines when to use each tool, allowing it to interpret the user's intent and invoke either the document_query_tool for retrieving document content or the document_comparison_tool for comparing documents. This process enables the AI to act autonomously, selecting and executing the correct function without needing explicit conditional logic.


## Task 4: Building a Quick End-to-End Prototype

https://huggingface.co/spaces/drewgenai/midterm_poc


## Task 5: Creating a Golden Test Data Set

The dataset is based on the submitted documents and the base model performed well across all metrics.

The base model is the Snowflake/snowflake-arctic-embed-m model.

### Base model evaluation 
| Metric                        | Value  |
|-------------------------------|--------|
| Context Recall                | 1.0000 |
| Faithfulness                  | 1.0000 |
| Factual Correctness           | 0.7540 |
| Answer Relevancy              | 0.9481 |
| Context Entity Recall         | 0.8095 |
| Noise Sensitivity Relevant    | 0.1973 |




## Task 6: Fine-Tuning Open-Source Embeddings
Link to fine tuning and testing
https://github.com/drewgenai/midterm_poc/blob/main/03-testembedtune.ipynb
link to fine tuned dataset
https://huggingface.co/drewgenai/midterm-compare-arctic-embed-m-ft



## Task 7: Assessing Performance

I ran the RAGAS evaluation on the finetuned model and the openai model as well. 

The finetuned model performed well across all metrics as well but not quite as well as the base Snowflake/snowflake-arctic-embed-m model where it didn't perform as well in context recall, but slightly in noise sensitivity.  

The openai model performed well across all metrics but not as well as the base Snowflake/snowflake-arctic-embed-m model, but with slightly worse noise sensitivity.  

### Finetuned Model Evaluation
| Metric                        | Value  |
|-------------------------------|--------|
| Context Recall                | 1.0000 |
| Faithfulness                  | 0.8500 |
| Factual Correctness           | 0.7220 |
| Answer Relevancy              | 0.9481 |
| Context Entity Recall         | 0.7917 |
| Noise Sensitivity Relevant    | 0.1111 |

### OpenAI Model Evaluation
| Metric                        | Value  |
|-------------------------------|--------|
| Context Recall                | 1.0000 |
| Faithfulness                  | 1.0000 |
| Factual Correctness           | 0.7540 |
| Answer Relevancy              | 0.9463 |
| Context Entity Recall         | 0.8095 |
| Noise Sensitivity Relevant    | 0.3095 |



The base model is the best performing model for the current use case.  

In the second half of the course I would like to explore having the application look for external standards to compare the protocols to further help with the comparison process.  I would also like it to evaluate the file type and use reasoning to determine the best approach to extracting the information and potentially accept more than 2 files.

There is a partially working agentic version of the application in app_working_on_agentic.py.  It has issues with the download links provided but works well with the agentic approach. Unfortunately the download link it provides is not working even though the file gets created properly. (https://github.com/drewgenai/midterm_poc/blob/main/app_working_on_agentic.py)  
* update note: the agentic version is now working as intended.  The download link is working and the file is created properly.

## Final Submission

1. GitHub: https://github.com/drewgenai/midterm_poc/blob/main/app.py  
2. GitHub: agentic poc https://github.com/drewgenai/midterm_poc/blob/main/app_working_on_agentic.py  
2. Public App link: https://huggingface.co/spaces/drewgenai/midterm_poc  
3. Public Fine-tuned embeddings: https://huggingface.co/drewgenai/midterm-compare-arctic-embed-m-ft  
4. loom link: https://www.loom.com/share/084d6c165917486097bcaea7deb12e88?sid=a5cc196f-76f1-4e18-bb92-ee61018f0b7e  