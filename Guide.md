# Hướng dẫn thực hành — Day 22: LangSmith + Prompt Versioning

Tài liệu này hướng dẫn từng bước chi tiết để hoàn thành lab. Đọc kỹ từng phần trước khi bắt tay vào code.

---

## Chuẩn bị (30 phút)

### Bước 1 — Tạo virtual environment

Luôn dùng virtual environment để tránh xung đột gói giữa các dự án:

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

Sau khi kích hoạt, dấu nhắc terminal sẽ hiển thị `(venv)` ở đầu dòng.

### Bước 2 — Cài thư viện

```bash
pip install -r requirements.txt
```

> Quá trình này có thể mất 5–10 phút lần đầu. Trong lúc chờ, hãy thực hiện bước tiếp theo.

### Bước 3 — Tạo tài khoản LangSmith và lấy API key

1. Truy cập [smith.langchain.com](https://smith.langchain.com) và đăng ký tài khoản miễn phí.
2. Vào **Settings** → **API Keys** → nhấn **Create API Key**.
3. Sao chép key (bắt đầu bằng `lsv2_`) — bạn sẽ cần dùng ở bước tiếp theo.
4. Tạo project mới tại **Projects** → **New Project**, đặt tên ví dụ `day22-lab`.

### Bước 4 — Cấu hình tệp `.env`

Sao chép tệp mẫu:

```bash
cp .env.example .env
```

Mở `.env` bằng bất kỳ editor nào và điền thông tin:

```env
# ─── LangSmith (bắt buộc) ─────────────────────────────────────────────────────
LANGSMITH_API_KEY=lsv2_pt_...        # API key vừa lấy ở trên
LANGSMITH_PROJECT=day22-lab          # Tên project trên LangSmith
LANGSMITH_WORKSPACE_ID=              # Optional workspace UUID
LANGCHAIN_TRACING_V2=true            # Bật tracing — không thay đổi giá trị này

# ─── Chọn provider LLM ────────────────────────────────────────────────────────
PROVIDER=openai                      # openai | gemini | anthropic | ollama | openrouter

# ─── OpenAI (nếu PROVIDER=openai) ────────────────────────────────────────────
OPENAI_API_KEY=sk-...

# ─── Google Gemini (nếu PROVIDER=gemini) ─────────────────────────────────────
GOOGLE_API_KEY=AIza...
GEMINI_MODEL=gemini-3.6-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001

# ─── Anthropic (nếu PROVIDER=anthropic) ──────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-...

# ─── OpenRouter (nếu PROVIDER=openrouter) ────────────────────────────────────
OPENROUTER_API_KEY=sk-or-...
```

Giải thích các biến quan trọng:
- `LANGSMITH_API_KEY`: Xác thực với LangSmith, không bao giờ chia sẻ công khai.
- `LANGSMITH_PROJECT`: Tên project để nhóm các traces lại, dễ tìm trên dashboard.
- `LANGCHAIN_TRACING_V2`: Phải là `true` để bật tracing — đặt sai sẽ mất toàn bộ traces.
- `PROVIDER`: Xác định LLM và embedding model nào được dùng trong toàn bộ lab.

### Bước 5 — Chọn LLM provider và điền thông tin

Chỉ cần điền thông tin cho provider bạn chọn, bỏ trống các provider còn lại:

- **OpenAI**: Ổn định nhất, khuyến nghị nếu có key. Dùng `gpt-4o-mini` để tiết kiệm chi phí.
- **Gemini**: Miễn phí với quota 15 request/phút — phù hợp nhưng có thể chậm hơn ở bước RAGAS.
- **Anthropic**: Chất lượng rất cao, chi phí trung bình.
- **Ollama**: Chạy hoàn toàn offline. Cần cài [ollama.ai](https://ollama.ai) và pull model trước (`ollama pull llama3.2`).
- **OpenRouter**: Tổng hợp nhiều model, có free tier cho một số model.

### Bước 6 — Xác minh cài đặt

```bash
cd src && python config.py
```

Nếu thấy thông báo xác nhận (không có lỗi đỏ), bạn đã sẵn sàng. Nếu có lỗi, kiểm tra lại các giá trị trong `.env`.

---

## Bước 1: RAG Pipeline với LangSmith (25–45 phút)

### Mục tiêu của bước này

Bạn sẽ xây dựng một RAG pipeline hoàn chỉnh: load dữ liệu → chunk → embed → index vào FAISS → tạo chain hỏi đáp → gắn decorator `@traceable` để mỗi câu hỏi tạo ra một trace trên LangSmith.

### Hướng dẫn từng bước

**1. Mở file `src/01_langsmith_rag_pipeline.py`**

Đọc qua toàn bộ file để hiểu cấu trúc trước khi bắt đầu viết code.

**2. Implement hàm `setup_vectorstore()`**

Hàm này cần thực hiện 4 việc theo thứ tự:

```python
def setup_vectorstore():
    embeddings = get_embeddings()                  # Lấy embedding model từ factory
    docs = load_knowledge_base()                   # Load text từ data/knowledge_base.txt
    chunks = split_text(docs)                      # Chia thành chunks nhỏ hơn
    vectorstore = build_vectorstore(chunks, embeddings)  # Tạo FAISS index
    return vectorstore
```

Gợi ý: hàm `get_embeddings()`, `load_knowledge_base()`, `split_text()`, và `build_vectorstore()` đã được implement trong `utils/`. Chỉ cần gọi đúng thứ tự.

**3. Định nghĩa `RAG_PROMPT`**

Tạo prompt template hướng dẫn LLM trả lời dựa trên context:

```python
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Bạn là trợ lý hữu ích. Chỉ trả lời dựa trên context được cung cấp. "
               "Nếu không tìm thấy thông tin, hãy nói 'Tôi không tìm thấy thông tin này'.\n\n"
               "Context:\n{context}"),
    ("human", "{question}"),
])
```

**4. Implement hàm `build_rag_chain()`**

Xây dựng LCEL chain nối retriever → prompt → LLM → parser:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
```

**5. Implement hàm `ask()` với decorator `@traceable`**

Đây là bước quan trọng nhất — decorator `@traceable` giúp mỗi lần gọi `ask()` tạo ra một trace trên LangSmith:

```python
from langsmith import traceable

@traceable(name="rag-query")
def ask(chain, question: str) -> str:
    return chain.invoke(question)
```

Lưu ý: `@traceable` phải được viết **ngay trên** định nghĩa hàm, không có dòng nào ở giữa.

**6. Hoàn thiện hàm `main()`**

```python
def main():
    print("Đang khởi tạo vector store...")
    vectorstore = setup_vectorstore()

    print("Đang xây dựng RAG chain...")
    chain = build_rag_chain(vectorstore)

    print(f"Đang chạy {len(QA_PAIRS)} câu hỏi qua RAG pipeline...")
    for i, (question, _) in enumerate(QA_PAIRS, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}] Q: {question[:60]}...")
        print(f"      A: {answer[:80]}...\n")

    print("Hoàn thành! Kiểm tra traces tại https://smith.langchain.com")
```

**7. Chạy bước 1**

```bash
python 01_langsmith_rag_pipeline.py
```

**8. Xác minh trên LangSmith**

Mở [smith.langchain.com](https://smith.langchain.com) → chọn project `day22-lab` → vào tab **Runs**. Bạn sẽ thấy ít nhất 50 traces, mỗi trace chứa câu hỏi, context được truy xuất và câu trả lời.

**9. Chụp ảnh màn hình bằng chứng**

Chụp màn hình giao diện LangSmith đang hiển thị danh sách traces → lưu vào `evidence/01_langsmith_traces.png`.

---

## Bước 2: Prompt Hub & A/B Routing (20–30 phút)

### Mục tiêu của bước này

Bạn sẽ tạo 2 system prompt có phong cách khác nhau, đẩy lên LangSmith Prompt Hub, pull về khi chạy, và định tuyến câu hỏi một cách tất định (cùng câu hỏi → luôn cùng prompt).

### Hướng dẫn từng bước

**1. Mở file `src/02_prompt_hub_ab_routing.py`**

**2. Đổi tên prompt thành tên của bạn**

Tìm các biến `PROMPT_V1_NAME` và `PROMPT_V2_NAME`, đổi thành tên duy nhất để tránh trùng với bạn khác:

```python
PROMPT_V1_NAME = "nguyen-van-a-rag-prompt-v1"   # Thay bằng tên của bạn
PROMPT_V2_NAME = "nguyen-van-a-rag-prompt-v2"
```

**3. Viết 2 system prompt với phong cách khác nhau**

Hai prompt phải có ngữ nghĩa rõ ràng khác nhau (không chỉ khác vài từ):

```python
# V1: Ngắn gọn, thân thiện
SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện. Trả lời ngắn gọn, rõ ràng dựa trên context. "
    "Nếu không có thông tin, hãy nói thẳng là không biết."
)

# V2: Chuyên nghiệp, có cấu trúc
SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích thông tin. Khi trả lời, hãy: "
    "1) Tóm tắt câu trả lời chính, "
    "2) Trích dẫn nguồn từ context, "
    "3) Nêu rõ mức độ chắc chắn của câu trả lời. "
    "Luôn dựa trên dữ liệu được cung cấp, không suy đoán thêm."
)
```

**4. Implement hàm `push_prompts_to_hub()`**

```python
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate

def push_prompts_to_hub():
    client = Client(api_key=LANGSMITH_API_KEY)

    template_v1 = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_V1 + "\n\nContext:\n{context}"),
        ("human", "{question}"),
    ])
    template_v2 = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_V2 + "\n\nContext:\n{context}"),
        ("human", "{question}"),
    ])

    client.push_prompt(PROMPT_V1_NAME, object=template_v1,
                       description="V1: Phong cách ngắn gọn, thân thiện")
    client.push_prompt(PROMPT_V2_NAME, object=template_v2,
                       description="V2: Phong cách chuyên nghiệp, có cấu trúc")
    print(f"Da push thanh cong: {PROMPT_V1_NAME} va {PROMPT_V2_NAME}")
```

**5. Implement hàm `pull_prompts_from_hub()`**

```python
def pull_prompts_from_hub():
    client = Client(api_key=LANGSMITH_API_KEY)
    prompt_v1 = client.pull_prompt(PROMPT_V1_NAME)
    prompt_v2 = client.pull_prompt(PROMPT_V2_NAME)
    return prompt_v1, prompt_v2
```

**6. Implement hàm `get_prompt_version()` — định tuyến bằng MD5 hash**

Hash MD5 của `request_id` đảm bảo cùng input luôn cho cùng output (tất định):

```python
import hashlib

def get_prompt_version(request_id: str) -> str:
    h = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return "v1" if h % 2 == 0 else "v2"
```

**7. Implement hàm `ask_ab()`**

```python
@traceable(name="ab-rag-query")
def ask_ab(chain, question: str, version_label: str) -> dict:
    answer = chain.invoke(question)
    return {
        "question": question,
        "answer": answer,
        "version": version_label,
    }
```

**8. Hoàn thiện hàm `main()`**

```python
def main():
    vectorstore = setup_vectorstore()
    prompt_v1, prompt_v2 = pull_prompts_from_hub()

    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    chain_v1 = build_chain(retriever, prompt_v1, llm)
    chain_v2 = build_chain(retriever, prompt_v2, llm)

    for i, (question, _) in enumerate(QA_PAIRS, 1):
        request_id = f"req-{i:03d}"
        version = get_prompt_version(request_id)
        chain = chain_v1 if version == "v1" else chain_v2

        result = ask_ab(chain, question, version)
        print(f"[{request_id}] [{result['version']}] Q: {question[:50]}...")
        print(f"             A: {result['answer'][:70]}...\n")
```

**9. Chạy và lưu log**

```bash
python 02_prompt_hub_ab_routing.py | tee ../evidence/02_ab_routing_log.txt
```

**10. Chụp ảnh Prompt Hub**

Vào [smith.langchain.com](https://smith.langchain.com) → **Prompt Hub** → tìm 2 prompt vừa push → chụp màn hình → lưu vào `evidence/02_prompt_hub.png`.

---

## Bước 3: RAGAS Evaluation (30–45 phút cài đặt + 15–30 phút chạy)

> **Lưu ý quan trọng:** Bước này cần 15–30 phút để chạy xong (đôi khi lâu hơn nếu dùng Gemini free tier). Hãy bắt đầu sớm và không đóng terminal trong lúc chờ.

### Mục tiêu của bước này

Đánh giá cả 2 phiên bản prompt trên 50 cặp QA bằng 4 chỉ số RAGAS và lưu kết quả vào JSON.

### Hướng dẫn từng bước

**1. Mở file `src/03_ragas_evaluation.py`**

**2. Sao chép `SYSTEM_V1` và `SYSTEM_V2` từ bước 2**

Đảm bảo 2 system prompt giống hệt bước 2 để kết quả có thể so sánh được.

**3. Implement hàm `run_rag()`**

Điểm quan trọng nhất: trường `contexts` trong RAGAS phải là `list[str]` (danh sách chuỗi), không phải một chuỗi ghép lại:

```python
def run_rag(chain, retriever, question: str) -> dict:
    # Lấy docs được truy xuất
    docs = retriever.invoke(question)

    # QUAN TRỌNG: contexts phải là list[str], không ghép thành một string
    contexts = [doc.page_content for doc in docs]

    # Lấy câu trả lời
    answer = chain.invoke(question)

    return {
        "answer": answer,
        "contexts": contexts,   # List các string riêng lẻ
    }
```

**4. Hoàn thiện hàm `collect_rag_outputs()`**

```python
def collect_rag_outputs(chain_v1, chain_v2, retriever):
    results_v1, results_v2 = [], []

    for i, (question, reference) in enumerate(QA_PAIRS, 1):
        print(f"  Đang xử lý câu {i}/{len(QA_PAIRS)}...", end="\r")

        out_v1 = run_rag(chain_v1, retriever, question)
        out_v2 = run_rag(chain_v2, retriever, question)

        results_v1.append({
            "question": question,
            "reference": reference,
            **out_v1,
        })
        results_v2.append({
            "question": question,
            "reference": reference,
            **out_v2,
        })

    return results_v1, results_v2
```

**5. Implement hàm `build_ragas_dataset()`**

`SingleTurnSample` yêu cầu đúng 4 trường này:

```python
from ragas import EvaluationDataset, SingleTurnSample

def build_ragas_dataset(results: list) -> EvaluationDataset:
    samples = [
        SingleTurnSample(
            user_input=r["question"],           # Câu hỏi
            response=r["answer"],               # Câu trả lời của LLM
            retrieved_contexts=r["contexts"],   # List[str] — contexts truy xuất được
            reference=r["reference"],           # Đáp án chuẩn từ qa_pairs.py
        )
        for r in results
    ]
    return EvaluationDataset(samples=samples)
```

**6. Implement hàm `run_ragas_eval()`**

```python
import warnings
warnings.filterwarnings("ignore")
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
import numpy as np

def run_ragas_eval(dataset: EvaluationDataset, llm_eval, emb_eval) -> dict:
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    result = evaluate(dataset, metrics=metrics, llm=llm_eval, embeddings=emb_eval)

    scores = {}
    for metric in metrics:
        name = metric.name
        scores[name] = float(np.mean(result[name]))

    return scores
```

**7. Hoàn thiện hàm `main()`**

```python
import json

def main():
    vectorstore = setup_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = get_llm()
    llm_eval = get_llm()       # LLM dùng cho RAGAS evaluation
    emb_eval = get_embeddings()

    chain_v1 = build_chain(retriever, SYSTEM_V1, llm)
    chain_v2 = build_chain(retriever, SYSTEM_V2, llm)

    print("Thu thap outputs tu 2 prompt versions...")
    results_v1, results_v2 = collect_rag_outputs(chain_v1, chain_v2, retriever)

    print("\nDang danh gia V1 (co the mat 10-20 phut)...")
    scores_v1 = run_ragas_eval(build_ragas_dataset(results_v1), llm_eval, emb_eval)

    print("\nDang danh gia V2 (co the mat 10-20 phut)...")
    scores_v2 = run_ragas_eval(build_ragas_dataset(results_v2), llm_eval, emb_eval)

    # In bảng so sánh
    print("\n" + "="*60)
    print(f"{'Chi so':<30} {'V1':>10} {'V2':>10}")
    print("-"*60)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        print(f"{metric:<30} {scores_v1[metric]:>10.4f} {scores_v2[metric]:>10.4f}")
    print("="*60)

    # Lưu báo cáo
    report = {"v1": scores_v1, "v2": scores_v2}
    with open("../data/ragas_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nDa luu bao cao vao data/ragas_report.json")
```

**8. Chạy bước 3**

```bash
python 03_ragas_evaluation.py
```

Quá trình sẽ mất 15–30 phút. Đừng đóng terminal.

**9. Chụp ảnh terminal**

Khi thấy bảng so sánh điểm xuất hiện → chụp màn hình → lưu vào `evidence/03_ragas_scores.png`.

**10. Sao chép báo cáo JSON vào thư mục evidence**

```bash
cp ../data/ragas_report.json ../evidence/03_ragas_report.json
```

---

## Bước 4: Guardrails AI Validators (20–30 phút)

### Mục tiêu của bước này

Xây dựng 2 validator tùy chỉnh: một để phát hiện và che thông tin cá nhân (PII), một để kiểm tra và sửa JSON lỗi từ đầu ra của LLM.

### Hướng dẫn từng bước

**1. Mở file `src/04_guardrails_validator.py`**

**2. Implement hàm `PIIDetector.validate()`**

Logic: duyệt qua danh sách pattern regex, tìm các match, thay thế bằng placeholder:

```python
import re
from guardrails.validators import PassResult, FailResult

# Các pattern PII cần phát hiện
PII_PATTERNS = {
    "EMAIL":       r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "PHONE":       r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
    "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
}

def validate(self, value, metadata):
    redacted_text = value
    found_pii = []

    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, redacted_text)
        if matches:
            found_pii.append(pii_type)
            # Thay thế tất cả match bằng placeholder
            redacted_text = re.sub(pattern, f"[{pii_type}_REDACTED]", redacted_text)

    if found_pii:
        return FailResult(
            error_message=f"Phat hien PII: {', '.join(found_pii)}",
            fix_value=redacted_text,
        )
    return PassResult()
```

**3. Implement hàm `JSONFormatter._repair()`**

Sửa các lỗi JSON phổ biến từ đầu ra LLM:

```python
def _repair(self, text: str) -> str:
    # Bước 1: Gỡ markdown code fences (```json ... ``` hoặc ``` ... ```)
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Bước 2: Sửa nháy đơn thành nháy đôi
    # Cẩn thận: chỉ thay nháy đơn dùng làm dấu ngoặc JSON
    text = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', text)  # Keys
    text = re.sub(r":\s*'([^']*)'", r': "\1"', text)     # String values

    # Bước 3: Xóa dấu phẩy thừa trước } hoặc ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    return text
```

**4. Implement hàm `JSONFormatter.validate()`**

```python
import json

def validate(self, value, metadata):
    # Thử parse trực tiếp
    try:
        json.loads(value)
        return PassResult()
    except json.JSONDecodeError:
        pass

    # Thử sửa rồi parse lại
    try:
        repaired = self._repair(value)
        json.loads(repaired)
        return PassResult(value_override=repaired)
    except json.JSONDecodeError:
        pass

    # Không thể sửa — trả về JSON lỗi chuẩn
    fallback = json.dumps({
        "error": "Khong the phan tich JSON",
        "raw": value[:200],
    }, ensure_ascii=False)
    return FailResult(
        error_message="Khong the sua JSON",
        fix_value=fallback,
    )
```

**5. Hoàn thiện hàm `demo_pii_guard()`**

```python
from guardrails import Guard, OnFailAction

def demo_pii_guard():
    # on_fail phải truyền vào constructor của validator, không phải Guard.use()
    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    test_cases = [
        "Xin chao, day la van ban binh thuong.",
        "Lien he qua email: user@example.com de biet them.",
        "So dien thoai cua toi la 555-123-4567.",
        "Ma so BHXH: 123-45-6789",
        "The tin dung: 4111 1111 1111 1111",
        "Email: a@b.com va so the 5500-0000-0000-0004 cung co day.",
    ]

    print("\n=== Demo PII Detector ===")
    for i, text in enumerate(test_cases, 1):
        result = guard.validate(text)
        status = "PASS" if result.validation_passed else "FIX"
        print(f"\n[Case {i}] {status}")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")
```

**6. Hoàn thiện hàm `demo_json_guard()`**

```python
def demo_json_guard():
    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        '{"name": "Alice", "age": 30}',                    # JSON hợp lệ
        '```json\n{"name": "Bob", "score": 95}\n```',      # Có fences
        "{'name': 'Charlie', 'active': true}",              # Nháy đơn
        '{"items": ["a", "b",], "total": 2,}',              # Phẩy thừa
        "day la van ban hoan toan khong phai JSON",          # Không thể sửa
    ]

    print("\n=== Demo JSON Formatter ===")
    for i, text in enumerate(test_cases, 1):
        result = guard.validate(text)
        status = "PASS" if result.validation_passed else "FIX/FAIL"
        print(f"\n[Case {i}] {status}")
        print(f"  Input:  {text[:60]}...")
        print(f"  Output: {result.validated_output}")
```

**7. Chạy và lưu log**

```bash
python 04_guardrails_validator.py | tee ../evidence/04_pii_demo_log.txt
```

---

## Kiểm tra trước khi nộp

Hãy đi qua danh sách này trước khi push lên GitHub:

**Mã nguồn:**
- [ ] `src/01_langsmith_rag_pipeline.py` chạy không có lỗi, in ra 50 câu hỏi/đáp
- [ ] `src/02_prompt_hub_ab_routing.py` chạy không có lỗi, log hiển thị nhãn v1/v2
- [ ] `src/03_ragas_evaluation.py` hoàn thành và in bảng điểm so sánh
- [ ] `src/04_guardrails_validator.py` chạy không có lỗi, hiển thị kết quả các test case
- [ ] `data/ragas_report.json` tồn tại và chứa điểm của cả V1 lẫn V2

**Bằng chứng (evidence):**
- [ ] `evidence/01_langsmith_traces.png` — Ảnh chụp LangSmith, thấy rõ ít nhất 50 traces
- [ ] `evidence/02_prompt_hub.png` — Ảnh chụp Prompt Hub, thấy rõ 2 prompt được đặt tên
- [ ] `evidence/02_ab_routing_log.txt` — File log có nội dung, hiển thị cả v1 lẫn v2
- [ ] `evidence/03_ragas_scores.png` — Ảnh chụp bảng điểm RAGAS trên terminal
- [ ] `evidence/03_ragas_report.json` — File JSON hợp lệ (sao chép từ `data/`)
- [ ] `evidence/04_pii_demo_log.txt` — File log có ít nhất 5 test case
- [ ] `evidence/04_json_demo_log.txt` — File log có ít nhất 4 test case

**Bảo mật:**
- [ ] `.gitignore` có dòng `.env`
- [ ] Không có API key nào xuất hiện trong bất kỳ tệp `.py` nào
- [ ] Lệnh `git diff --cached` không hiển thị nội dung `.env`

---

## Nộp bài

### 1. Tạo GitHub repository public

Truy cập [github.com/new](https://github.com/new), tạo repository mới:
- Tên: `day22-langsmith-lab` (hoặc tên tự chọn)
- Visibility: **Public** (bắt buộc để chấm điểm)
- Không tích "Initialize with README" (vì bạn đã có code)

### 2. Push code lên GitHub

```bash
# Tạo .gitignore nếu chưa có
echo ".env" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "venv/" >> .gitignore

git init
git add .
git status   # Kiểm tra lần cuối — đảm bảo KHÔNG thấy .env trong danh sách

git commit -m "Day 22: LangSmith + Prompt Versioning lab"
git remote add origin https://github.com/<tên-của-bạn>/day22-langsmith-lab.git
git push -u origin main
```

### 3. Nộp thông tin

Nộp qua cổng của khóa học:

1. **URL GitHub repository** — ví dụ: `https://github.com/nguyen-van-a/day22-langsmith-lab`
2. **URL LangSmith project** — ví dụ: `https://smith.langchain.com/o/<org-id>/projects/p/<project-id>`
3. Xác nhận thư mục `evidence/` đã có đầy đủ 7 tệp

> **Nhắc lại:** Không bao giờ commit tệp `.env` hoặc dán API key vào mã nguồn. Đây là lỗi vi phạm bảo mật bị trừ 10 điểm tự động.
