# Lab 2 — Advanced RAG Results (CS 5542)

> **Course:** CS 5542 — Big Data Analytics and Apps  
> **Lab:** Advanced RAG Systems Engineering  
> **Student Name:**  Quynh (Tina) Nguyen  
> **GitHub Username:**  tinana2k  
> **Date:**  01/29/2025

---

## 1. Project Dataset
- **Domain:**  Car Repair Fraud   
- **# Documents:**  the 8 txt files (car repair fraud, credit card, telemarketing frauds,....)  
- **Data Source (URL / Description):**  https://github.com/tinana2k/Comp-Sci-5542-Tina-Nguyen/tree/main/Week_2/project_data 
- **Why this dataset fits my use case:**  
  This dataset consists of real-world fraud scenarios related to car repair, credit card, and telemarketing fraud, which represent a common and practical problem domain.  
  The documents contain overlapping concepts, exceptions, and ambiguous situations, making them well-suited for evaluating retrieval accuracy, re-ranking, and grounded generation in an advanced RAG system.  
  Using these texts reflects realistic user queries where precise evidence retrieval is critical to avoid misinformation.

  
---

## 2. Queries + Mini Rubric

### Q1
- **Query:** What is car repair fraud?
- **Relevant Evidence (keywords / entities / constraints):**
  - Definition of car repair fraud
  - Mentions of deceptive repair practices
  - Examples such as unnecessary repairs or inflated charges
- **Correct Answer Criteria (1–2 bullets):**
  - Clearly defines car repair fraud in plain language
  - Includes at least one concrete example from the documents

---

### Q2
- **Query:** What are common warning signs that a consumer may be a victim of car repair fraud?
- **Relevant Evidence (keywords / entities / constraints):**
  - Red flags such as vague estimates, pressure tactics, or refusal to provide written invoices
  - Mentions of consumer complaints or scam indicators
  - Any listed preventative advice or detection methods
- **Correct Answer Criteria (1–2 bullets):**
  - Lists multiple warning signs supported by document evidence
  - Explains how these signs indicate fraudulent behavior

---

### Q3 (Ambiguous / Edge Case)
- **Query:** Is charging extra for additional repairs always considered car repair fraud?
- **Relevant Evidence (keywords / entities / constraints):**
  - Mentions of legitimate vs fraudulent repair practices
  - Conditions where additional charges may be allowed
  - Language such as “may,” “depends,” or “unless authorized”
- **Correct Answer Criteria (1–2 bullets):**
  - Explicitly acknowledges that the answer depends on circumstances
  - States that additional information or authorization is required, or notes insufficient evidence
 

---

## 3. System Design

- **Chunking Strategy:** Semantic (paragraph/sentence-based) + Fixed (character-based baseline)
- **Chunk Size / Overlap:** Fixed = ~1200 chars with ~200-char overlap; Semantic = up to ~1400 chars per chunk
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store / Index:** FAISS (IndexFlatIP with normalized embeddings for cosine similarity)
- **Keyword Retriever:** BM25
- **Hybrid α Value(s):** 0.2, 0.5, 0.8 (tested)
- **Re-ranking Method:** Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)

### Design Rationale
I used both fixed and semantic chunking to compare how chunk boundaries affect retrieval quality and answer grounding. 
FAISS with a lightweight embedding model provides fast semantic retrieval, while BM25 improves precision for exact fraud-related keywords and named entities.
Hybrid retrieval with multiple α values helps balance lexical matching and semantic similarity, which is important because fraud documents often share similar vocabulary across different scam types.
Finally, cross-encoder re-ranking improves the ordering of top candidates so the generator uses the strongest evidence first, trading a small amount of latency for better accuracy.


---

## 4. Results

| Query | Method | Precision@5 | Recall@10 |
|------|--------|-------------|-----------|
| Q1 | Hybrid (Semantic) | 0.60 | 1.00 |
| Q2 | Hybrid (Semantic) | 0.40 | 0.80 |
| Q3 | Hybrid (Semantic) | 0.20 | 0.40 |

<img width="381" height="83" alt="image" src="https://github.com/user-attachments/assets/55785a58-4baa-4d76-a235-eb5e53bdba7e" />
<img width="1658" height="510" alt="image" src="https://github.com/user-attachments/assets/600a0d0b-94ff-4e89-b53c-f1f64869ea90" />


---

## 5. Failure Case

- **What failed?**  
  The system struggled to answer the ambiguous query about whether additional repair charges always constitute car repair fraud. The retrieved evidence included both legitimate repair practices and fraudulent scenarios, which made it difficult to produce a definitive answer.

- **Which layer failed?**  
  Retrieval.

- **Proposed system-level fix:**  
  Introduce finer-grained semantic chunking or metadata-based filtering (e.g., separating legitimate repair guidelines from fraud case descriptions) to reduce conflicting evidence during retrieval.


---

## 6. Evidence of Grounding

Provide one example of a **RAG-grounded answer with citations**:

> **Answer:**  
> Car repair fraud occurs when a mechanic or repair shop intentionally deceives a customer by performing unnecessary repairs, inflating costs, or misrepresenting the condition of a vehicle. Common examples include charging for repairs that were never performed or exaggerating the severity of mechanical issues to justify higher fees.

> **Citations:** [Chunk 1], [Chunk 3]

---

## 7. Reflection (3–5 Sentences)
What did you learn about **system design vs. model performance** when building this RAG pipeline?

This project showed that system design choices such as chunking strategy, hybrid retrieval, and re-ranking often have a greater impact on RAG performance than the underlying language model alone. Even with a strong embedding model, poor chunk boundaries or retrieval imbalance can significantly reduce answer quality. Hybrid retrieval and re-ranking improved precision by combining lexical and semantic signals. Overall, building an effective RAG system requires careful engineering of each pipeline stage rather than relying solely on model capability.

---

## Reproducibility Checklist
- [Yes ] Project dataset included or linked  
- [ Yes] Queries + rubric filled  
- [Yes ] Results table completed  
- [ Yes] Screenshots included in repo  
- [ Yes] Notebook runs end-to-end  

---

> *CS 5542 — UMKC School of Science & Engineering*
