# Lab Requirements — Day 22: LangSmith + Prompt Versioning

## Python Version
Python 3.10 or higher

## Install All Dependencies

```bash
pip install -r requirements.txt
```

## requirements.txt

```
langchain==1.3.14
langchain-core==1.5.3
langchain-openai==1.4.1
langchain-community==0.3.31
dataclasses-json==0.6.7
marshmallow>=3.18.0,<4.0.0
typing-inspect>=0.7.1,<1.0.0
mypy-extensions>=0.3.0
langchain-text-splitters==1.1.2
langchain-google-genai==4.3.2
langchain-anthropic==1.5.4
langchain-ollama==1.1.0
langsmith>=0.10.0,<1.0.0
openai>=2.49.0,<3.0.0
faiss-cpu>=1.7.0
ragas==0.4.3
guardrails-ai==0.10.2
python-dotenv>=1.0.0
tiktoken>=0.5.0
datasets>=4.0.0,<5.0.0
pyarrow>=21.0.0,<22.0.0
numpy>=1.25.0,<3.0.0
```

## Package Purposes

| Package | Used For |
|---------|---------|
| `langchain` | Core LLM framework |
| `langchain-openai` | ChatOpenAI, OpenAIEmbeddings |
| `langchain-community` | FAISS vectorstore integration |
| `langchain-text-splitters` | RecursiveCharacterTextSplitter |
| `langsmith` | LangSmith tracing, Prompt Hub client |
| `openai` | Direct OpenAI API calls |
| `faiss-cpu` | Similarity search index |
| `ragas` | RAG evaluation metrics |
| `guardrails-ai` | Output validation framework |
| `python-dotenv` | Load `.env` file |
| `tiktoken` | Token counting for text splitters |
| `datasets` | Required by RAGAS internally |
| `pyarrow` | Arrow backend required by the pinned datasets/RAGAS stack |
| `numpy` | Averaging RAGAS score lists |

## Important Version Notes

### RAGAS 0.4.x
- Use `from ragas.metrics import faithfulness, answer_relevancy, ...` (NOT from `ragas.metrics.collections`)
- `result[metric_name]` returns a **list** of floats for multiple samples — use `numpy.mean()` to average
- Pass `llm=` and `embeddings=` to the `evaluate()` function, not to metric constructors

### Guardrails AI 0.10.x
- `on_fail` parameter belongs in the **validator constructor**: `MyValidator(on_fail=OnFailAction.FIX)`
- `Guard.use()` accepts validator **instances**, not classes
- `Guard.validate(text)` is the main entry point

### LangChain 1.x
- Use `ChatOpenAI(api_key=..., base_url=..., model=...)` for custom endpoints
- Use `OpenAIEmbeddings(api_key=..., base_url=..., model=...)` for custom embedding endpoints

## Environment Variables

Copy this to your `.env` file:


> ⚠️ **Never commit `.env` to git.** Add it to `.gitignore`.

## Verify Installation

Run the config check:
```bash
python config.py
```

Expected output:
```
✅ Config loaded successfully
   LangSmith project : your-project-name
   OpenAI endpoint   : https://...
   Default LLM model : gpt-5.4-mini
   Embedding model   : text-embedding-3-small
```
