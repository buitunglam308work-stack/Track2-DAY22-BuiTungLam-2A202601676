# Chào mừng các bạn đến với Day 22: LangSmith + Prompt Versioning

## Tổng quan

Trong lab này, bạn sẽ xây dựng một hệ thống hỏi đáp hoàn chỉnh tích hợp nhiều công nghệ AI hiện đại:

- **RAG Pipeline**: Xây dựng pipeline Retrieval-Augmented Generation sử dụng FAISS làm vector store và LangChain để kết nối các thành phần.
- **LangSmith Tracing**: Theo dõi và quan sát toàn bộ luồng xử lý của ứng dụng LLM thông qua LangSmith dashboard.
- **Prompt Hub & A/B Testing**: Quản lý phiên bản prompt trên LangSmith Prompt Hub và thực hiện A/B routing để so sánh hiệu quả giữa các phiên bản.
- **RAGAS Evaluation**: Đánh giá chất lượng hệ thống RAG theo 4 chỉ số định lượng: faithfulness, answer relevancy, context recall, context precision.
- **Guardrails AI**: Triển khai các bộ kiểm duyệt tự động để phát hiện thông tin cá nhân (PII) và sửa lỗi định dạng JSON trong đầu ra của LLM.

---

## Mục tiêu học tập

Sau khi hoàn thành lab này, bạn sẽ có thể:

- Xây dựng và triển khai RAG pipeline hoàn chỉnh với LangChain LCEL và FAISS vector store.
- Tích hợp LangSmith để theo dõi, gỡ lỗi và phân tích hiệu suất của ứng dụng LLM trong thực tế.
- Quản lý vòng đời prompt bằng LangSmith Prompt Hub và thực hiện A/B testing có kiểm soát.
- Đánh giá hệ thống RAG một cách định lượng bằng framework RAGAS với các chỉ số chuẩn công nghiệp.
- Áp dụng Guardrails AI để xây dựng validator tùy chỉnh nhằm bảo vệ đầu ra của LLM khỏi dữ liệu nhạy cảm và lỗi định dạng.

---

## Yêu cầu trước

Trước khi bắt đầu, hãy đảm bảo bạn đã có:

- **Python 3.10 trở lên** — kiểm tra bằng lệnh `python --version`
- **API key** của ít nhất một trong các nhà cung cấp LLM sau:
  - OpenAI (`OPENAI_API_KEY`)
  - Google Gemini (`GOOGLE_API_KEY`)
  - Anthropic Claude (`ANTHROPIC_API_KEY`)
  - OpenRouter (`OPENROUTER_API_KEY`)
  - Ollama (chạy local, không cần API key)
- **Tài khoản LangSmith** — đăng ký miễn phí tại [smith.langchain.com](https://smith.langchain.com) và lấy API key

---

## Cài đặt môi trường

### 1. Cài thư viện

```bash
pip install -r requirements.txt
```

> Lần đầu cài có thể mất 5–10 phút do nhiều gói phụ thuộc.

### 2. Cấu hình tệp `.env`

Sao chép tệp mẫu và điền thông tin của bạn:

```bash
cp .env.example .env
```

Mở tệp `.env` và điền các giá trị sau:

```env
# LangSmith — bắt buộc cho tất cả các bước
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=day22-lab
LANGCHAIN_TRACING_V2=true
LANGSMITH_WORKSPACE_ID=              # Optional workspace UUID

# Chọn một trong các provider bên dưới
PROVIDER=openai

# OpenAI (nếu dùng PROVIDER=openai)
OPENAI_API_KEY=sk-...

# Google Gemini (nếu dùng PROVIDER=gemini)
GOOGLE_API_KEY=AIza...
GEMINI_MODEL=gemini-3.6-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001

# Anthropic (nếu dùng PROVIDER=anthropic)
ANTHROPIC_API_KEY=sk-ant-...

# OpenRouter (nếu dùng PROVIDER=openrouter)
OPENROUTER_API_KEY=sk-or-...
```

### 3. Chọn LLM provider

Đặt biến `PROVIDER` trong `.env` thành một trong các giá trị sau:

| Giá trị      | Nhà cung cấp      | Ghi chú                         |
|--------------|-------------------|---------------------------------|
| `openai`     | OpenAI GPT        | Mặc định, ổn định nhất          |
| `gemini`     | Google Gemini     | Miễn phí với quota giới hạn     |
| `anthropic`  | Anthropic Claude  | Chất lượng cao                  |
| `ollama`     | Ollama (local)    | Không cần API key, cần GPU/CPU  |
| `openrouter` | OpenRouter        | Tổng hợp nhiều model            |

### 4. Xác minh cài đặt

```bash
cd src && python config.py
```

Nếu không có lỗi xuất hiện, bạn đã sẵn sàng bắt đầu.

---

## Cấu trúc dự án

```
Lab/
├── src/
│   ├── config.py                      # Tải .env, cấu hình providers
│   ├── utils/
│   │   ├── llm_factory.py             # Factory tạo LLM và Embeddings (5 providers)
│   │   └── data_loader.py             # Load knowledge base, chunk, build FAISS
│   ├── qa_pairs.py                    # 50 cặp câu hỏi + đáp án chuẩn
│   ├── 01_langsmith_rag_pipeline.py   # Bước 1: RAG + LangSmith tracing
│   ├── 02_prompt_hub_ab_routing.py    # Bước 2: Prompt Hub + A/B routing
│   ├── 03_ragas_evaluation.py         # Bước 3: RAGAS evaluation (~15-30 phút)
│   ├── 04_guardrails_validator.py     # Bước 4: Guardrails AI validators
│   └── run_all.py                     # Chạy tất cả các bước
├── data/
│   ├── knowledge_base.txt             # Tài liệu nguồn cho RAG
│   └── ragas_report.json              # Được tạo ra ở Bước 3
├── evidence/                          # Nộp thư mục này lên GitHub
│   ├── 01_langsmith_traces.png
│   ├── 02_prompt_hub.png
│   ├── 02_ab_routing_log.txt
│   ├── 03_ragas_scores.png
│   ├── 03_ragas_report.json
│   ├── 04_pii_demo_log.txt
│   └── 04_json_demo_log.txt
├── .env.example                        # Template biến môi trường
├── requirements.txt
├── README.md
├── rubric.md
└── Guide.md
```

---

## Các nhiệm vụ

Lab được chia thành 4 nhiệm vụ, mỗi nhiệm vụ 25 điểm (tổng 100 điểm):

| Nhiệm vụ | Tên                              | Điểm | Thời gian ước tính   |
|----------|----------------------------------|------|----------------------|
| 1        | RAG Pipeline với LangSmith       | 25đ  | 25–45 phút           |
| 2        | Prompt Hub & A/B Routing         | 25đ  | 20–30 phút           |
| 3        | RAGAS Evaluation                 | 25đ  | 45–75 phút           |
| 4        | Guardrails AI Validators         | 25đ  | 20–30 phút           |

**Nhiệm vụ 1 — RAG Pipeline với LangSmith (25đ):** Xây dựng vector store từ knowledge base, tạo RAG chain, và tích hợp `@traceable` để ghi lại ít nhất 50 traces trên LangSmith dashboard.

**Nhiệm vụ 2 — Prompt Hub & A/B Routing (25đ):** Soạn 2 system prompt có ngữ nghĩa khác biệt, đẩy lên LangSmith Prompt Hub, pull về khi chạy, và định tuyến câu hỏi theo hash của `request_id`.

**Nhiệm vụ 3 — RAGAS Evaluation (25đ):** Chạy 50 cặp QA qua cả 2 phiên bản prompt, xây dựng `EvaluationDataset`, tính 4 chỉ số RAGAS, và đạt faithfulness ≥ 0.8 với ít nhất 1 phiên bản.

**Nhiệm vụ 4 — Guardrails AI Validators (25đ):** Triển khai `PIIDetector` tự động che thông tin cá nhân và `JSONFormatter` tự động sửa JSON lỗi từ đầu ra của LLM.

---

## Chạy lab

### Chạy từng bước riêng lẻ

```bash
cd src

# Bước 1: RAG Pipeline với LangSmith tracing
python 01_langsmith_rag_pipeline.py

# Bước 2: Prompt Hub và A/B routing
python 02_prompt_hub_ab_routing.py

# Bước 3: RAGAS evaluation (mất 15–30 phút)
python 03_ragas_evaluation.py

# Bước 4: Guardrails AI validators
python 04_guardrails_validator.py
```

### Chạy toàn bộ lab

```bash
cd src && python run_all.py
```

### Chạy một bước cụ thể

```bash
cd src && python run_all.py --step 3
```

---

## Nộp bài

### 1. Tạo GitHub repository

Tạo repository public mới trên GitHub với tên ví dụ `day22-langsmith-lab`.

### 2. Thu thập bằng chứng (evidence)

Đảm bảo thư mục `evidence/` chứa đầy đủ 7 tệp sau:

```
evidence/
├── 01_langsmith_traces.png      ← Ảnh chụp màn hình LangSmith dashboard (≥ 50 traces)
├── 02_prompt_hub.png            ← Ảnh chụp màn hình Prompt Hub (2 phiên bản)
├── 02_ab_routing_log.txt        ← Output console của bước 2
├── 03_ragas_scores.png          ← Ảnh chụp terminal hiển thị điểm RAGAS
├── 03_ragas_report.json         ← Báo cáo JSON từ RAGAS
├── 04_pii_demo_log.txt          ← Output console của PII detector
└── 04_json_demo_log.txt         ← Output console của JSON formatter
```

### 3. Lưu output console vào tệp

Sử dụng lệnh `tee` để vừa in ra màn hình vừa lưu vào tệp:

```bash
python script.py | tee evidence/output.txt
```

Ví dụ cụ thể:

```bash
python 02_prompt_hub_ab_routing.py | tee ../evidence/02_ab_routing_log.txt
python 04_guardrails_validator.py  | tee ../evidence/04_pii_demo_log.txt
```

### 4. Push lên GitHub và nộp

```bash
git init
git add .
git commit -m "Day 22: LangSmith + Prompt Versioning lab submission"
git remote add origin https://github.com/<tên-của-bạn>/day22-langsmith-lab.git
git push -u origin main
```

Nộp URL GitHub repository và URL LangSmith project của bạn qua cổng nộp bài của khóa học.

---

## Tips và lưu ý

**LangSmith tracing — đặt biến môi trường đúng thứ tự:**
Các biến `LANGCHAIN_TRACING_V2`, `LANGSMITH_API_KEY`, và `LANGSMITH_PROJECT` phải được đặt **trước khi import bất kỳ thứ gì từ LangChain**. Nếu import trước khi đặt biến, tracing sẽ không hoạt động.

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"   # Phải đặt trước
os.environ["LANGSMITH_API_KEY"]    = "..."     # Phải đặt trước
from langchain_core.prompts import ChatPromptTemplate  # Sau đó mới import
```

**RAGAS chậm — bắt đầu sớm:**
Bước 3 sẽ mất từ 15 đến 30 phút để hoàn thành do phải gọi LLM cho mỗi sample trong bộ đánh giá. Hãy bắt đầu bước này ngay khi bước 2 xong, đặc biệt nếu bạn đang dùng model có rate limit thấp.

**Guardrails AI — `on_fail` phải truyền đúng chỗ:**
Tham số `on_fail` phải được truyền vào **constructor của validator**, không phải vào `Guard.use()`:

```python
# ĐÚNG
Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

# SAI — sẽ không hoạt động đúng
Guard().use(PIIDetector(), on_fail=OnFailAction.FIX)
```

**Bảo mật — không bao giờ commit `.env`:**
Tệp `.env` chứa API key nhạy cảm. Đảm bảo `.gitignore` đã có dòng `.env` trước khi push lên GitHub. Chỉ commit tệp `.env.example` (không chứa giá trị thật). Vi phạm quy tắc này sẽ bị trừ 10 điểm tự động.

---

## Tài liệu tham khảo

| Tài liệu                    | Đường dẫn                                                          |
|-----------------------------|--------------------------------------------------------------------|
| LangSmith Docs              | https://docs.smith.langchain.com                                   |
| LangChain LCEL              | https://python.langchain.com/docs/concepts/lcel                    |
| LangSmith Prompt Hub        | https://docs.smith.langchain.com/prompt-hub                        |
| RAGAS Documentation         | https://docs.ragas.io                                              |
| Guardrails AI               | https://www.guardrailsai.com/docs                                  |
| FAISS (Facebook AI)         | https://faiss.ai                                                   |
| LangChain FAISS Integration | https://python.langchain.com/docs/integrations/vectorstores/faiss  |
