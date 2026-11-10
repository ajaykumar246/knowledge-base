# RAG-Powered Knowledge Base API: Technical Interview Preparation Guide

This guide is designed to help you prepare for technical discussions, system design interviews, and engineering manager reviews regarding this project's architecture, trade-offs, and design choices.

---

## 1. Core Architectural Concepts

### What is RAG (Retrieval-Augmented Generation)?
* **The Problem with Raw LLMs:** Large Language Models (LLMs) are trained on static historical datasets. They lack real-time access to private enterprise documents and are susceptible to **hallucinations** (confidently fabricating incorrect facts).
* **RAG vs. Fine-Tuning:**
  * **Fine-Tuning** teaches a model *how to speak* (style, tone, specialized syntax) by updating its internal weights. It is computationally expensive, slow to update with new data, and prone to catastrophic forgetting.
  * **RAG** acts as an *open-book exam*. Instead of retraining the model, RAG searches an external vector database for relevant text passages and injects them directly into the LLM's prompt at query time.
* **The 3 Stages of RAG:**
  1. **Ingestion:** Documents are loaded, split into manageable semantic chunks, converted into dense numerical vectors (embeddings), and stored in a vector index.
  2. **Retrieval:** When a user asks a question, the query is embedded into the same vector space, and the database computes vector similarity (e.g., Cosine Similarity or Euclidean Distance) to find the top-$k$ most relevant chunks.
  3. **Generation:** The retrieved chunks are placed alongside the user query into a structured system prompt, and the LLM generates a grounded answer citing the provided sources.

---

### Why use FastAPI `BackgroundTasks` for Document Ingestion?
* **Non-Blocking ASGI Event Loop:** FastAPI runs on an asynchronous event loop (`asyncio`). CPU-bound tasks (extracting text from a 50-page PDF and calculating neural network embeddings) or synchronous disk I/O block the event loop. If run synchronously inside the request handler, all other concurrent requests to the API would freeze until the PDF finished processing.
* **User Experience (Immediate Feedback):** Using `BackgroundTasks`, the `POST /upload` endpoint performs quick validations, saves the raw file stream, enqueues the worker function, and immediately returns HTTP `202 Accepted` with a `task_id`. The client is not kept waiting on an open HTTP socket for 10-30 seconds.
* **When `BackgroundTasks` is Appropriate vs. Distributed Queues:**
  * *FastAPI `BackgroundTasks`:* Ideal for single-node apps, lightweight background operations, and low-to-medium throughput where process restarts rarely happen.
  * *Limitations:* If the server process crashes or restarts, pending in-memory background tasks are lost. In heavy production enterprise workloads, this should be upgraded to a distributed task queue (e.g., **Celery** or **ARQ** backed by Redis or RabbitMQ).

---

### How does Redis Store Conversation Memory?
* **Stateless API Architecture:** Modern backend microservices must be stateless so that multiple API replicas can run behind a load balancer without sticky sessions.
* **Key-Value Schema:** Each conversation session is isolated using a namespaced key:
  ```text
  chat_history:{session_id}
  ```
* **Redis List Data Structure:**
  * Messages are appended chronologically using `RPUSH chat_history:{session_id} '{"role": "human", "content": "..."}'`.
  * The full history or last $N$ turns are fetched using `LRANGE chat_history:{session_id} 0 -1`.
* **Automatic Expiration via TTL:** Every time a new message is appended, the key's TTL is refreshed (`EXPIRE chat_history:{session_id} 86400`). This ensures active chats stay alive while inactive sessions automatically expire after 24 hours, preventing Redis RAM exhaustion.

---

### Agentic Routing vs. Static RAG Pipelines
* **The Problem with Naive RAG Pipelines:** A traditional RAG chain blindly executes a vector search for every input. If a user simply types *"Hi, good morning!"*, a naive pipeline runs a vector similarity search, retrieves irrelevant document chunks, burns unnecessary embedding computations, and feeds garbage context to the LLM.
* **Agentic Decision Making (Tool Calling):** By configuring the LLM as an Agent equipped with a `Search_Documents` tool, the model inspects user intent first:
  * **General greetings / chitchat:** Handled directly with zero vector database calls.
  * **Domain-specific / factual questions:** The agent invokes `Search_Documents`, reads the returned context, and formulates a grounded answer with citations.

---

## 2. Common Engineering Manager & Technical Interview Questions

### Q1: Why did you choose local Hugging Face embeddings (`all-MiniLM-L6-v2`) instead of commercial APIs like OpenAI's `text-embedding-3-small`?

> **Answer:**  
> "I chose `sentence-transformers/all-MiniLM-L6-v2` based on three engineering trade-offs:
> 1. **Data Privacy & Compliance:** The documents never leave the local application environment during embedding generation, which is a common hard requirement for HIPAA, GDPR, or proprietary internal enterprise documents.
> 2. **Cost & Rate Limits:** Local embeddings incur zero external API costs and are never subject to third-party rate limits (HTTP 429) or cloud downtime.
> 3. **Dimensionality & Latency:** `all-MiniLM-L6-v2` produces compact 384-dimensional vectors (compared to OpenAI's 1536 dimensions), reducing ChromaDB disk footprint and accelerating vector distance calculations by up to 4x on CPU.
> 
> *Production Graduation Note:* For multilingual documents or applications requiring deep cross-lingual semantic nuance, I would benchmark against `text-embedding-3-small` or open-source BGE/E5 models."

---

### Q2: What happens if an uploaded PDF is 500 pages long? What are the failure points in this architecture, and how would you scale it?

> **Answer:**  
> "In the current architecture, processing a 500-page PDF via FastAPI `BackgroundTasks` exposes three vulnerabilities:
> 1. **Worker Process Contention:** Embedding 500 pages involves significant CPU matrix operations that could starve the Python process, degrading API response latency for other endpoints.
> 2. **Memory Spikes:** Loading large PDFs into memory simultaneously can trigger out-of-memory (OOM) errors in containerized environments.
> 3. **Loss of Task on Crash:** If the container restarts midway through page 250, the progress is lost because `BackgroundTasks` lives only in the process memory.
> 
> **How to Scale in Production:**
> - Offload ingestion to a distributed task queue like **Celery** or **ARQ** with Redis/RabbitMQ.
> - Process pages in streaming batches (e.g., 20 pages per sub-task) to keep memory bounded.
> - Track ingestion state in PostgreSQL (`status: PENDING | PROCESSING | COMPLETED | FAILED`) with retry policies and dead-letter queues (DLQ)."

---

### Q3: How did you select the chunk size (1000 characters) and chunk overlap (150 characters)? What happens if they are tuned improperly?

> **Answer:**  
> "Chunk size and overlap represent a fundamental trade-off between **semantic specificity** and **contextual completeness**:
> - **If chunk size is too small (e.g., 100 chars):** Sentences get chopped in half. The vector embedding loses its broader semantic meaning, and retrieved chunks lack sufficient context for the LLM to understand the surrounding discussion.
> - **If chunk size is too large (e.g., 5000 chars):** Chunks combine multiple disparate topics into a single vector, diluting the embedding's focus. Furthermore, large chunks exhaust the LLM's context window and increase prompt token costs.
> - **Why 1000 characters with 150 overlap:** 1000 characters corresponds to approximately 200–250 words (a coherent paragraph). A 150-character (~15%) overlap ensures that boundary sentences are not severed, preserving semantic continuity between adjacent chunks."

---

### Q4: How do you mitigate hallucinations and ensure the Agent provides faithful citations?

> **Answer:**  
> "We implement a defense-in-depth approach across prompt engineering, retrieval scoring, and model parameters:
> 1. **Prompt Grounding Constraints:** The system prompt explicitly instructs the agent: *'Synthesize answers strictly based on the retrieved context. If the context does not contain the answer, explicitly state that the information is missing.'*
> 2. **Deterministic Temperature (`temperature=0.0`):** Setting temperature to zero minimizes stochastic creativity and forces the model to select highest-probability, context-grounded tokens.
> 3. **Metadata Attachment:** During chunking, we attach `source`, `chunk_index`, and `task_id` into each Chroma chunk. The search tool injects these headers into the retrieved passages, allowing the LLM to directly cite `[Source: document.pdf | Chunk: 3]` in its final response.
> 4. **Future Enhancement:** In enterprise production, I would add a self-consistency check or hallucination evaluator (such as RAGAS or TruLens) to compute faithfullness and answer relevance scores before returning the payload to the user."

---

### Q5: In a production Kubernetes cluster running 5 replicas of this API, how does conversation memory remain consistent?

> **Answer:**  
> "Because conversation memory is stored in an external, centralized **Redis** instance rather than in Python process memory, our FastAPI application remains completely **stateless**:
> - Request 1 from User A might hit Pod #1, which writes the user's message to Redis key `chat_history:session-123`.
> - Request 2 from User A might be routed by the Kubernetes Ingress to Pod #4. Pod #4 reads `chat_history:session-123` from Redis and immediately has the full historical context.
> - This decoupling allows the API pods to scale horizontally up and down on CPU/RAM metrics via a Kubernetes Horizontal Pod Autoscaler (HPA) without dropping conversation state."

---

### Q6: Why ChromaDB for local vector storage instead of PostgreSQL with `pgvector` or managed Pinecone? When would you migrate?

> **Answer:**  
> "ChromaDB with local directory persistence (`PersistentClient`) was selected for this project because it provides an embedded, self-contained vector database requiring zero external infrastructure setup—making it perfect for local development, fast automated testing, and standalone edge deployments.
> 
> **When to Migrate:**
> - **PostgreSQL (`pgvector`):** If the application already uses Postgres for relational user data, orders, and permissions, migrating to `pgvector` eliminates an extra database from the stack and allows unified ACID transactions and hybrid SQL + vector queries.
> - **Pinecone / Qdrant / Milvus:** When scaling past 5–10 million vectors with strict requirements for distributed sharding, high-availability replication, and sub-50ms p99 retrieval latencies under thousands of queries per second."

---

### Q7: What telemetry and observability metrics would you add before deploying this API to production?

> **Answer:**  
> "I would instrument the application across three key observability pillars:
> 1. **System & Infrastructure Metrics (Prometheus + Grafana):**
>    - HTTP request latency (p50, p95, p99) per endpoint.
>    - Background task queue lag and execution duration.
>    - Redis memory utilization and connection pool saturation.
> 2. **RAG-Specific Telemetry (LangSmith or OpenInference / Arize Phoenix):**
>    - Embedding model inference latency per chunk.
>    - Top-$k$ vector similarity distance distribution (alerting if similarity drops below a confidence threshold).
>    - Tool routing frequency (percentage of queries routing to `Search_Documents` vs direct conversational responses).
>    - LLM token consumption, prompt-to-completion token ratios, and estimated API expenditure per session."
