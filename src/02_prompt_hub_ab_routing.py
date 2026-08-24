"""
Bước 2 — Prompt Hub & A/B Routing
===================================
NHIỆM VỤ:
  1. Viết 2 system prompt khác nhau (V1: ngắn gọn, V2: có cấu trúc)
  2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
  3. Pull lại từ Hub qua client.pull_prompt()
  4. Implement A/B routing tất định: hash(request_id) % 2 → V1 hoặc V2
  5. Chạy 50 câu hỏi qua router → ≥ 50 LangSmith traces nữa

DELIVERABLE: 2 prompt version hiển thị trong Prompt Hub trên https://smith.langchain.com
"""
import sys
import hashlib
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.console import configure_utf8_console

configure_utf8_console()

import config  # ⚠️ phải import trước LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable
from langsmith.utils import LangSmithConflictError

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from utils.retry import invoke_with_retry
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
PROMPT_V1_NAME = "buitunglam-day22-rag-prompt-v1"
PROMPT_V2_NAME = "buitunglam-day22-rag-prompt-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện. Chỉ dùng context được cung cấp để trả lời "
    "ngắn gọn, rõ ràng trong 2-4 câu. Nếu context không có câu trả lời, "
    "hãy nói rằng bạn không tìm thấy thông tin.\n\nContext:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích thông tin. Chỉ sử dụng các sự thật "
    "được nêu trực tiếp trong context; không suy đoán hoặc bổ sung kiến thức bên ngoài. "
    "Trình bày ngắn gọn theo cấu trúc: kết luận trực tiếp, sau đó 1-2 sự thật "
    "hỗ trợ từ context (tối đa 4 câu). Không thêm nhận định về độ chắc chắn, "
    "bình luận meta, hàm ý, hoặc khuyến nghị. Nếu context không đủ, chỉ nêu thông tin "
    "nào còn thiếu.\n\nContext:\n{context}"
)

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def push_prompts_to_hub(client: Client, strict: bool = True):
    """
    Upload cả 2 prompt templates lên LangSmith Prompt Hub.
    Gợi ý: client.push_prompt(name, object=template, description="...")
    """
    try:
        url = client.push_prompt(
            PROMPT_V1_NAME,
            object=PROMPT_V1,
            description="V1 - grounded, friendly, concise answers",
        )
        print(f"✅ Đã push V1 → {url}")
    except LangSmithConflictError as e:
        if "Nothing to commit" not in str(e):
            raise
        print("✅ V1 trên Hub đã là phiên bản mới nhất (không có thay đổi)")
    except Exception as e:
        print(f"⚠️  V1 lỗi: {e}")
        if strict:
            raise RuntimeError("Prompt Hub push failed for V1") from e

    try:
        url = client.push_prompt(
            PROMPT_V2_NAME,
            object=PROMPT_V2,
            description="V2 - grounded, structured expert answers",
        )
        print(f"✅ Đã push V2 → {url}")
    except LangSmithConflictError as e:
        if "Nothing to commit" not in str(e):
            raise
        print("✅ V2 trên Hub đã là phiên bản mới nhất (không có thay đổi)")
    except Exception as e:
        print(f"⚠️  V2 lỗi: {e}")
        if strict:
            raise RuntimeError("Prompt Hub push failed for V2") from e


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client, allow_local_fallback: bool = False) -> dict:
    """
    Tải 2 prompt từ LangSmith Prompt Hub.
    Fallback về template local nếu Hub không khả dụng.

    Gợi ý: client.pull_prompt(name) → ChatPromptTemplate

    Trả về: {name: ChatPromptTemplate}
    """
    prompts = {}

    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Đã pull '{PROMPT_V1_NAME}' từ Hub")
    except Exception:
        if not allow_local_fallback:
            raise RuntimeError(f"Prompt Hub pull failed for '{PROMPT_V1_NAME}'")
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V1_NAME}'")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Đã pull '{PROMPT_V2_NAME}' từ Hub")
    except Exception:
        if not allow_local_fallback:
            raise RuntimeError(f"Prompt Hub pull failed for '{PROMPT_V2_NAME}'")
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V2_NAME}'")

    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Xác định prompt version dựa trên MD5 hash của request_id.

    Quy tắc: hash chẵn → PROMPT_V1_NAME | hash lẻ → PROMPT_V2_NAME
    TÍNH CHẤT: cùng request_id LUÔN cho cùng kết quả (deterministic).

    Gợi ý:
        hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME
    """
    hash_int = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16)

    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Chạy RAG chain với prompt version được chọn bởi router.

    Bước:
      a) Retrieve top-3 docs từ retriever
      b) Ghép page_content thành context string
      c) Chạy (prompt | llm | StrOutputParser()).invoke({"context": ..., "question": ...})
      d) Trả về {"question": ..., "answer": ..., "version": ...}
    """
    docs = invoke_with_retry(
        lambda: retriever.invoke(question),
        label=f"Step 2 {version} retrieval",
    )

    context = "\n\n".join(doc.page_content for doc in docs)

    chain = prompt | llm | StrOutputParser()
    answer = invoke_with_retry(
        lambda: chain.invoke({"context": context, "question": question}),
        label=f"Step 2 {version} query",
    )

    return {
        "question": question,
        "answer": answer,
        "version": version,
        "context": context,
    }


# ── 7. Setup Vectorstore (tái sử dụng logic Bước 1) ───────────────────────
def setup_vectorstore():
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 8. Main ────────────────────────────────────────────────────────────────
def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Prompt Hub A/B routing")
    parser.add_argument(
        "--allow-local-fallback",
        action="store_true",
        help="allow local prompts only for offline development; graded mode is strict Hub-only",
    )
    args = parser.parse_args(argv)

    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)
    if not config.validate_langsmith_connection():
        sys.exit(1)

    client = config.get_langsmith_client()

    push_prompts_to_hub(client, strict=not args.allow_local_fallback)

    prompts = pull_prompts_from_hub(client, allow_local_fallback=args.allow_local_fallback)

    # Tạo vectorstore, retriever và LLM
    vectorstore = setup_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = get_llm()

    # Chạy A/B routing cho tất cả câu hỏi
    v1_count, v2_count = 0, 0
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"

        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        result = ask_ab(retriever, llm, prompt, question, version_tag)

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1
        print(f"[{i+1:02d}] [{request_id}] [prompt-{version_tag}] {question[:55]}...")
        print(f"          A: {str(result['answer'])[:100]}...")

    print(f"\n📊 Routing: V1={v1_count} câu | V2={v2_count} câu | Tổng={len(SAMPLE_QUESTIONS)}")
    print("✅ Bước 2 hoàn thành! Kiểm tra Prompt Hub và traces trên LangSmith.")


if __name__ == "__main__":
    main()
