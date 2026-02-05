# CS 5542 — Lab 3: Multimodal RAG Systems & Retrieval Evaluation

**Course:** CS 5542 — Big Data Analytics and Applications    
**Lab:** Multimodal RAG Systems & Retrieval Evaluation        
**Student Name:** Tina (Quynh) Nguyen     
**GitHub Username:** tinana2k     
**Date:** 02/05/2026       

---

## 1. Executive Summary
This project implements a **Multimodal Retrieval-Augmented Generation (RAG) & Retrieval Evaluation** system capable of ingesting PDF documents and images (charts/diagrams) to answer complex domain-specific queries. The system utilizes:
* **Hybrid Retrieval:** Combining Sparse (TF-IDF) and Dense (SentenceTransformers/FAISS) vector search.
* **Multimodal Ingestion:** Extracting text from PDFs via `PyMuPDF` and semantic content from images via **OCR (Tesseract)**.
* **Generation:** Comparing a lightweight extractive baseline, a local LLM (TinyLlama), and a cloud API (Gemini).

The experiments demonstrate that **Multimodal RAG** significantly improves answer coverage for queries relying on visual data (flowcharts, maps) compared to text-only baselines.

### Repository Structure

```text
COMP_SCI_5542/
├── Week_3/                           # Multimodal RAG Systems (Lab 3)
│   ├── project_data_mm/              # Knowledge Base
│   │   ├── figures/                  # Extracted charts and diagrams
│   │   │   ├── CCPAvsGDPR-Nov.png
│   │   │   └── ...
│   │   ├── doc1.pdf
│   │   ├── doc2.pdf
│   │   └── ...
│   ├── project_report/                       # Reports/Screenshots for Week 3 modules
│   │       ├── ingestion_report.jpg
│   │       └── ...
│   ├── project_src/                          # Source code for Week 3 modules
│   │   └── CS5542_Lab3.ipynb
│   ├── README_MULTIMODAL_RAG.md                    # Lab 3 specific documentation
│   └── requirements.txt              # Dependencies (PyMuPDF, Tesseract, etc.)
└── README.md                         # Project documentation and roadmap
```

---

## 2. Project Dataset
- **Domain:** Bank Fraud/Policies 
- **# Documents:**	5 pdf files and 10 images (scanned pages, flowchart, tables, figures and screenshots etc.)
- **Data Source (URL / Description):** [tinana2k Github Repository](https://github.com/tinana2k/Comp-Sci-5542-Tina-Nguyen/tree/main/Week_3/project_data_mm)
---

## 3. Queries + Rubrics

The following table outlines the three test queries used to evaluate the RAG system, along with the keyword-based ground truth rubric used for automated scoring.

### **Query 1: Credit Card Fraud Red Flags and Staff Response**

- **ID:** Q1  
- **Question:** Based on bank fraud policies and the red-flags checklist, what are the most common red flags associated with credit card fraud, and what immediate actions should bank staff take when these indicators are detected?

| Category | Keywords (Ground Truth) |
|---------|--------------------------|
| **Must Have** | `credit card fraud`, `red flags`, `unusual transactions`, `account monitoring`, `staff escalation`, `fraud investigation` |
| **Optional** | `identity theft`, `transaction velocity`, `policy guidance`, `risk management`, `fraud statistics` |

---

### **Query 2: Internal Control Weaknesses and Fraud Risk**

- **ID:** Q2  
- **Question:** Using the Internal Control Red Flags checklist and related bank fraud policy documents, identify two internal control weaknesses that increase the risk of fraud and describe one control or mitigation strategy that can reduce this risk.

| Category | Keywords (Ground Truth) |
|---------|--------------------------|
| **Must Have** | `internal control weaknesses`, `lack of segregation of duties`, `insufficient oversight`, `fraud risk`, `mitigation controls` |
| **Optional** | `policy enforcement`, `monitoring procedures`, `compliance timelines`, `risk management framework` |

---

### **Query 3: Suspicious Activity Reporting (SAR) Thresholds**

- **ID:** Q3  
- **Question:** According to bank fraud policy documents and supporting figures, what dollar thresholds or conditions trigger the requirement to file a Suspicious Activity Report (SAR), and why are these thresholds important for fraud detection and regulatory compliance?

| Category | Keywords (Ground Truth) |
|---------|--------------------------|
| **Must Have** | `Suspicious Activity Report`, `SAR threshold`, `dollar amount`, `fraud detection`, `regulatory compliance` |
| **Optional** | `transaction monitoring`, `fraud trends`, `risk indicators`, `financial institutions`, `reporting obligations` |

---

## 4. Methodology

### A. Data Ingestion & OCR
* **Text:** Extracted from PDFs (`doc1.pdf` - `doc5.pdf`) using `PyMuPDF`.
* **Images:** 10 figures (charts, maps) processed using **Tesseract OCR**. Captions were combined with OCR text to create searchable "Image Items."
* **Chunking Strategy:**
    * *Baseline:* Page-based chunking.
    * *Ablation:* Fixed-size sliding window chunking (`CHUNK_SIZE=900`, `OVERLAP=150`).

### B. Retrieval Strategies
We evaluated five retrieval configurations:
1.  **Sparse Only:** TF-IDF (keyword matching).
2.  **Dense Only:** `all-MiniLM-L6-v2` embeddings + FAISS (semantic matching).
3.  **Hybrid:** Weighted fusion of Sparse + Dense scores (`Alpha=0.5`).
4.  **Hybrid + Rerank:** Re-scoring top candidates using a Cross-Encoder (`ms-marco-MiniLM`).
5.  **Multimodal:** Hybrid Text retrieval + Sparse Image retrieval.

### C. Generators
1.  **Extractive:** Returns top-2 retrieved snippets (Baseline).
2.  **Local LLM:** `TinyLlama-1.1B-Chat` (Quantized, runs on T4 GPU).
3.  **Cloud API:** Google `gemini-2.5-flash` (Ground Truth/Gold Standard).

---

## 5. Evaluation Results

### A. Ingestion Report following Track B
![ingestion_report](https://github.com/tinana2k/Comp-Sci-5542-Tina-Nguyen/blob/main/Week_3/project_report/ingestion_report.jpg)

### B. Lightweight Generator
![generator_lightweight](https://github.com/tinana2k/Comp-Sci-5542-Tina-Nguyen/blob/main/Week_3/project_report/generator_lightweight.jpg)

### C. LLM API Call Generator
![generator_API_Call](https://github.com/tinana2k/Comp-Sci-5542-Tina-Nguyen/blob/main/Week_3/project_report/generator_API_Call.jpg)

### D. LLM Local Generator
<img width="1737" height="768" alt="image" src="https://github.com/user-attachments/assets/00e96195-d2da-45fc-83ab-8700645d2ca1" />

<img width="1337" height="182" alt="image" src="https://github.com/user-attachments/assets/a6451e16-14fb-41bd-aeda-30d5221caa2e" />

### E. Quantitative Retrieval Metrics
*=== Final Deliverable Table (Query x Method x Metrics) ===*

<img width="801" height="670" alt="image" src="https://github.com/user-attachments/assets/0e07d0bf-cce3-41ba-b788-7a8a1aee5382" />


### F. Manual Evaluation Metrics for Generator Models
![evaluation_metrics](https://github.com/tinana2k/Comp-Sci-5542-Tina-Nguyen/blob/main/Week_3/project_report/full_evaluation_metrics_all_models.jpg)

### G. Ablation Study: Text-Only vs. Multimodal
*Does adding images help answer the questions?*

<img width="1707" height="510" alt="image" src="https://github.com/user-attachments/assets/15437da6-314f-473a-b22b-a37a7f6d82d9" />


**Conclusion:** Multimodal RAG is essential for Q1 and Q2, as the text chunks alone described the *rules* but the images contained the *workflow* and *geographic distribution*.

---

## 6. Failure Analysis

### **1. Documented Failure Case**

**Query:**
**Q3** — *“What is the exact dollar threshold for filing a Suspicious Activity Report (SAR) according to these documents and figures?”*

**Observed Failure:**
The system produced a partially relevant answer discussing fraud reporting in general but **failed to consistently identify the exact SAR dollar threshold**. In some cases, the response was vague or relied on indirect language rather than explicitly stating the numeric threshold required for SAR filing.

---

### **2. Root Cause Analysis**

This issue is primarily a **retrieval failure**, not a generation failure.

1. **Numeric Detail Loss:**
   The SAR threshold is a **specific numeric value**, which appears infrequently in the documents. Sparse and dense retrievers tend to prioritize semantic context (e.g., “fraud reporting,” “compliance”) over exact numbers, causing the key threshold value to be missed.

2. **Chunking Around Tables and Timelines:**
   The SAR threshold information is likely embedded in **tables, timelines, or compliance charts** (e.g., NACHA or fraud compliance figures). Fixed-size chunking may have separated the **numeric threshold** from the surrounding explanatory text, preventing the retriever from capturing the full rule in a single chunk.

---

### **3. Proposed Concrete Fix**

**Improvement:**
Enhance retrieval for numeric and policy-specific facts.

* Increase **chunk overlap** to ensure numeric values remain attached to their explanatory context.
* Add **keyword boosting or regex-based retrieval** for monetary patterns (e.g., “$”, “USD”, “threshold”) to improve recall of exact values.
* Retrieve a larger candidate set (e.g., top-20) before **reranking**, allowing the cross-encoder to surface precise compliance rules.

These changes would improve the system’s ability to retrieve and correctly report exact regulatory thresholds.
---

## 7. How to Run This Project

1.  **Install Dependencies:**
    ```bash
    pip install PyMuPDF sentence-transformers faiss-cpu reportlab pytesseract
    sudo apt-get install tesseract-ocr  # If running on Linux/Colab
    ```
2.  **Data Setup:**
    Ensure `project_data_mm/` exists. If not, the notebook will automatically download the dataset or generate sample data.
3.  **Execution:**
    Run all cells in `CS5542_Lab3.ipynb`.
    * **Note:** To run the LLM Generator without an API key, rely on the `TinyLlama` (Local) section.

---

## Reproducibility Checklist Yes(Y) or No(N)
- [Y] Project dataset included or linked  
- [Y] Queries + rubric filled  
- [Y] Results table completed  
- [Y] Screenshots included in repo  
- [Y] Notebook runs end-to-end  

---

> *CS 5542 — UMKC School of Science & Engineering*
