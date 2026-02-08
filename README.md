# Causal Analysis and Interactive Reasoning over Conversational Data

## 📋 Project Overview

This project implements an intelligent system that analyzes conversational transcripts to extract causal relationships and provide interactive reasoning about customer service issues. The system uses semantic embeddings, similarity search, and sentiment analysis to understand why certain events occur in customer conversations and engages in context-aware interactive dialogue.

### Key Capabilities

- **Causal Explanation (Task 1)**: Analyze conversational transcripts to extract why certain events occur
- **Interactive Reasoning (Task 2)**: Engage in multi-turn conversations with memory of previous queries
- **Semantic Search**: Find semantically similar conversations using FAISS and sentence embeddings
- **Sentiment Analysis**: Detect frustration and emotional signals in conversations
- **Evidence-Based Reasoning**: Support conclusions with specific examples from the transcript database

---

## 🎯 Features

- **Semantic Similarity Search**: Uses `sentence-transformers` (all-MiniLM-L6-v2) to find related conversations
- **FAISS-Based Indexing**: Efficient vector similarity search on conversational embeddings
- **Causal Signal Extraction**: Identifies repetition, frustration, and agent delays
- **Follow-up Detection**: Automatically detects related queries using semantic similarity
- **Context Memory**: Maintains conversation context across multiple queries
- **Evidence-Based Explanations**: Provides supporting examples from real conversations

---

## 📦 System Requirements

### Software Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **RAM**: Minimum 8GB (16GB recommended for optimal performance)
- **Disk Space**: At least 2GB for dependencies and data

### Python Dependencies
```
pandas >= 1.3.0
numpy >= 1.21.0
sentence-transformers >= 2.2.0
faiss-cpu >= 1.7.0
nltk >= 3.6.0
scikit-learn >= 1.0.0
```

---

## 🚀 Installation & Setup

### Step 1: Clone or Download the Repository

```bash
cd /path/to/IIT\ BBSR\ HACKATHON
```

### Step 2: Create a Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies

The notebook includes an auto-installation cell, but you can also install manually:

```bash
pip install -q pandas numpy sentence-transformers faiss-cpu nltk scikit-learn
```

### Step 4: Verify Installation

```bash
python -c "import faiss, sentence_transformers, nltk; print('All dependencies installed successfully!')"
```

---

## 📊 Data Structure & Preprocessing

### Input Data Files

#### 1. **Conversational_Transcript_Dataset.json** (Main Dataset)
Contains customer service conversation transcripts with the following structure:

```json
{
  "transcripts": [
    {
      "transcript_id": "6794-8660-4606-3216",
      "time_of_interaction": "2025-10-03 20:22:00",
      "domain": "E-commerce & Retail",
      "intent": "Delivery Investigation",
      "reason_for_call": "Description of the issue",
      "conversation": [
        {
          "speaker": "Agent",
          "text": "Agent's response"
        },
        {
          "speaker": "Customer",
          "text": "Customer's message"
        }
      ]
    }
  ]
}
```

**Key Fields:**
- `transcript_id`: Unique identifier for each conversation
- `conversation`: Array of speaker-text pairs (Agent/Customer turns)
- `intent`: The nature of the customer's issue
- `domain`: Industry/service domain

#### 2. **query_dataset.csv** (Test Queries)
Contains queries to be processed by Task 1 and Task 2:

```csv
query_id,query_text,expected_intent,task_type,is_followup,notes
Q1,Why do customers escalate complaints?,complaint_escalation,task1,no,Primary causal explanation query
```

**Columns:**
- `query_id`: Unique query identifier
- `query_text`: The actual query text
- `expected_intent`: Expected intent category
- `task_type`: Either "task1" (causal) or "task2" (interactive)
- `is_followup`: Boolean indicating if this is a follow-up query
- `notes`: Description of the query

### Data Preprocessing Steps

The notebook automatically handles the following preprocessing:

1. **Loading Transcripts**: Parse JSON and extract conversation structure
2. **Text Aggregation**: Concatenate all turns in each conversation
3. **Embedding Generation**: Convert text to semantic vectors (384-dimensional)
4. **Index Creation**: Build FAISS index for similarity search
5. **Signal Extraction**: Identify causal indicators (keywords, sentiment scores)

---

## 🧠 Model Architecture & Components

### 1. **Semantic Embedding Model**
- **Model Name**: `all-MiniLM-L6-v2` (from Hugging Face)
- **Output Dimension**: 384
- **Purpose**: Convert conversational text to semantic vectors
- **Initialization**:
  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer("all-MiniLM-L6-v2")
  ```

### 2. **FAISS Vector Index**
- **Index Type**: `IndexFlatL2` (L2 Euclidean distance)
- **Purpose**: Efficient similarity search over embeddings
- **Initialization**:
  ```python
  import faiss
  dimension = embeddings.shape[1]
  index = faiss.IndexFlatL2(dimension)
  index.add(np.array(embeddings))
  ```

### 3. **Sentiment Analysis**
- **Tool**: VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Source**: NLTK library
- **Compound Score Threshold**: < -0.3 for negative sentiment
- **Usage**: Detect frustration in customer messages

### 4. **Causal Signal Extraction**

**Signals Extracted:**
- **Repetition**: Count of complaint keywords (delay, problem, issue, waiting, etc.)
- **Frustration**: Count of segments with negative sentiment
- **Agent Delay**: Difference between agent and customer message counts

```python
complaint_keywords = [
    "delay", "problem", "issue", "again", "still",
    "not received", "waiting", "angry", "frustrated"
]
```

### 5. **Follow-up Detection**
- **Mechanism**: Semantic similarity between current and previous query
- **Threshold**: 0.7 cosine similarity
- **Purpose**: Determine if context memory should be reused

---

## ▶️ Running the Application

### Method 1: Interactive Jupyter Notebook (Recommended)

1. **Install Jupyter** (if not already installed):
   ```bash
   pip install jupyter
   ```

2. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

3. **Open the Notebook**:
   - Navigate to `causal_analysis_and_interactive_reasoning_over_conversational_data.ipynb`
   - Click to open it in Jupyter

4. **Execute Cells in Order**:
   - **Cell 1**: Install dependencies
   - **Cell 2**: Import libraries
   - **Cell 3**: Load transcript dataset
   - **Cell 4**: Preprocess conversations
   - **Cell 5**: Generate embeddings
   - **Cell 6**: Create FAISS index
   - **Cell 7**: Define retrieval function
   - **Cell 8-10**: Define causal signal extraction functions
   - **Cell 11**: Run Task 1 (Causal Explanation)
   - **Cell 12-15**: Initialize interactive reasoning functions
   - **Cell 16**: Load query dataset
   - **Cell 17**: Execute all tasks (Task 1 & Task 2)

5. **View Results**:
   - Output appears in cell output areas
   - Each query produces formatted explanations with evidence

### Method 2: Command-Line Execution (Advanced)

Create a Python script `run_analysis.py`:

```python
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download required NLTK data
nltk.download('vader_lexicon')

# Load data
with open("Conversational_Transcript_Dataset.json") as f:
    data = json.load(f)

# Create DataFrame
records = []
for t in data["transcripts"]:
    text = ""
    for turn in t["conversation"]:
        text += f"{turn['speaker']}: {turn['text']} "
    records.append({
        "transcript_id": t["transcript_id"],
        "intent": t["intent"],
        "text": text
    })

df = pd.DataFrame(records)

# Generate embeddings and create index
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Load and process queries
query_df = pd.read_csv("query_dataset.csv")

for _, row in query_df.iterrows():
    print(f"\nQUERY: {row['query_text']}")
    # Processing logic here
```

Run with:
```bash
python run_analysis.py
```

---

## 📁 Project Structure

```
IIT BBSR HACKATHON/
├── README.md                                          # This file
├── causal_analysis_and_interactive_reasoning_over_conversational_data.ipynb
│   ├── Dependencies Installation
│   ├── Library Imports
│   ├── Data Loading
│   ├── Data Preprocessing
│   ├── Embedding Generation
│   ├── FAISS Index Creation
│   ├── Retrieval Functions
│   ├── Causal Signal Extraction
│   ├── Causal Explanation Engine
│   ├── Task 1 Output
│   ├── Interactive Reasoning Setup
│   ├── Response Formatting
│   └── Task 2 Execution
├── Conversational_Transcript_Dataset.json           # Main transcript data
├── query_dataset.csv                                 # Test queries
└── .github/
    └── instructions/
        └── kluster-code-verify.instructions.md
```

---

## 🔧 Usage Examples

### Example 1: Task 1 - Causal Explanation

**Input Query:**
```
"Why do customers escalate complaints?"
```

**Output:**
```
================================================================================
ASSISTANT:
I understand your question:
→ "Why do customers escalate complaints?"

Here's what's happening based on similar conversations:

WHY THIS EVENT OCCURS:
• Repeated unresolved customer issues
• Escalating customer frustration
• Delays or lack of resolution by agents

CAUSAL FLOW:
Repeated issue → Frustration → Escalation

SUPPORTING EXAMPLES:
Transcript ID: 6794-8660-4606-3216
Customer: I'm frustrated about waiting...
Agent: Let me look into this...
...
Confidence: Supported by 4 similar transcripts
================================================================================
```

### Example 2: Task 2 - Interactive Reasoning with Memory

**Query 1:**
```
"Why do customers escalate complaints?"
```

**Query 2 (Follow-up):**
```
"Why does this happen so frequently?"
```

**System Detection:**
- Recognizes Query 2 as follow-up (similarity > 0.7)
- Reuses context from Query 1
- Provides focused response without repeating context

---

## 🔄 Workflow

### Complete Execution Pipeline

```
1. Load Conversational Transcripts (JSON)
    ↓
2. Preprocess & Aggregate Conversations
    ↓
3. Generate Semantic Embeddings
    ↓
4. Create FAISS Index for Fast Search
    ↓
5. Load Query Dataset (CSV)
    ↓
6. For Each Query:
    ├─ If Task 1 (Causal):
    │  ├─ Retrieve similar conversations
    │  ├─ Extract causal signals
    │  ├─ Generate explanation with evidence
    │  └─ Format and display
    │
    └─ If Task 2 (Interactive):
       ├─ Check if follow-up (similarity > 0.7)
       ├─ Reuse or retrieve new context
       ├─ Generate explanation
       ├─ Format for chat-style display
       └─ Update memory for next query
```

---

## 📈 Model Components Explained

### Causal Signal Aggregation

1. **Repetition Score**
   - Counts occurrences of complaint keywords
   - Threshold: > 3 indicates repeated issues
   
2. **Frustration Score**
   - Counts negative sentiment segments (compound < -0.3)
   - Threshold: > 2 indicates escalating frustration

3. **Agent Delay Score**
   - Measures agent responsiveness
   - Calculated as: `Agent turns - Customer turns`
   - Threshold: > 2 indicates potential delays

### Confidence Calculation

```
Confidence = Number of similar transcripts found
             (default: 4 most similar conversations)
```

---

## 🐛 Troubleshooting

### Issue 1: Module Not Found Error
**Error**: `ModuleNotFoundError: No module named 'faiss'`

**Solution**:
```bash
pip install faiss-cpu
# or for GPU support
pip install faiss-gpu
```

### Issue 2: Memory Error During Embedding Generation
**Error**: `MemoryError` or system freeze

**Solution**:
- Reduce batch size in embedding generation
- Use a machine with more RAM
- Process transcripts in chunks

```python
# Process in batches
batch_size = 100
embeddings = []
for i in range(0, len(df), batch_size):
    batch = df["text"].iloc[i:i+batch_size].tolist()
    batch_emb = model.encode(batch, show_progress_bar=True)
    embeddings.extend(batch_emb)
```

### Issue 3: NLTK Data Not Downloaded
**Error**: `Resource vader_lexicon not found`

**Solution**:
```python
import nltk
nltk.download('vader_lexicon')
```

### Issue 4: JSON Decoding Error
**Error**: `json.JSONDecodeError`

**Solution**:
- Ensure `Conversational_Transcript_Dataset.json` is valid JSON
- Check file encoding (should be UTF-8)
- Verify file is not corrupted

### Issue 5: Query Results Show No Evidence
**Problem**: Queries return low confidence results

**Solution**:
- Ensure query matches domain of conversations
- Increase retrieval count `k` parameter
- Check sentiment analysis threshold
- Verify FAISS index was populated correctly

---

## 📊 Expected Output Files

The system generates console output. To save results, you can modify the notebook:

```python
# Save Task 1 results
with open("task1_results.txt", "w") as f:
    f.write(explanation)

# Save Task 2 results
with open("task2_results.txt", "w") as f:
    f.write(formatted_response)
```

---

## 🎓 Understanding Causal Analysis

### What is Causal Analysis?

This system identifies **why** certain patterns occur in conversations:
- Why do customers escalate?
- What causes frustration?
- Why do issues repeat?

### How Does It Work?

1. **Find Similar Cases**: Uses semantic similarity to locate relevant conversations
2. **Extract Signals**: Identifies frustration, repetition, and delays
3. **Build Narrative**: Explains the causal chain
4. **Support with Evidence**: Provides exact quotes from transcripts

### Limitations

- Analysis based on conversational patterns, not ground truth causality
- Requires sufficient similar examples in dataset
- Sentiment analysis may not capture all emotional nuances
- Domain-specific terms may not be fully understood

---

## 🔐 Data Privacy & Security

- All data processing happens locally on your machine
- No data is sent to external servers
- Embeddings are computed and stored locally using FAISS
- Consider data sensitivity if sharing results

---

## 📝 Citation & References

**Libraries Used:**
- Sentence Transformers (https://www.sbert.net/)
- FAISS (https://ai.meta.com/tools/faiss/)
- NLTK (https://www.nltk.org/)
- Pandas (https://pandas.pydata.org/)

---

### Running into Issues?

1. Check the **Troubleshooting** section above
2. Verify all dependencies are installed
3. Ensure data files are in the correct location
4. Check file permissions and encoding

### Improving the System

Potential enhancements:
- Add more sophisticated causal inference models
- Implement multi-language support
- Create REST API for remote access
- Add visualization dashboards
- Implement feedback loops for model improvement

---

## 📄 License

This project is part of the IIT BBSR Hackathon submission.

---

## 🎯 Quick Start Checklist

- [ ] Clone/download the repository
- [ ] Create and activate virtual environment
- [ ] Install dependencies: `pip install -q pandas numpy sentence-transformers faiss-cpu nltk scikit-learn`
- [ ] Verify `Conversational_Transcript_Dataset.json` is in the project folder
- [ ] Verify `query_dataset.csv` is in the project folder
- [ ] Open Jupyter Notebook: `jupyter notebook`
- [ ] Run notebook cells in order (1-17)
- [ ] Review output for Task 1 and Task 2 results
- [ ] Check results in console output

---


---

**Last Updated**: February 2026
**Python Version**: 3.8+
**Status**: Production Ready
