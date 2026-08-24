"""
Tiện ích để tải và xử lý dữ liệu cho RAG pipeline.

Cách dùng:
    from utils.data_loader import load_knowledge_base, split_text, build_vectorstore

    text        = load_knowledge_base()
    chunks      = split_text(text, chunk_size=500, chunk_overlap=50)
    vectorstore = build_vectorstore(chunks, embeddings)
"""
import hashlib
from pathlib import Path

from utils.retry import invoke_with_retry


_VECTORSTORE_CACHE = {}


def embed_documents_in_batches(
    texts: list[str],
    embeddings,
    batch_size: int = 50,
    attempts: int = 5,
    sleep=None,
) -> list[list[float]]:
    """Embed documents in quota-friendly batches with bounded retry/backoff."""
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        retry_kwargs = {}
        if sleep is not None:
            retry_kwargs["sleep"] = sleep
        batch_vectors = invoke_with_retry(
            lambda: embeddings.embed_documents(batch),
            attempts=attempts,
            base_delay=1.0,
            label=f"Embedding batch {start + 1}-{start + len(batch)}",
            **retry_kwargs,
        )
        vectors.extend(batch_vectors)
    if len(vectors) != len(texts):
        raise ValueError(f"Embedding count mismatch: got {len(vectors)}, expected {len(texts)}")
    return vectors


def load_knowledge_base(path: str = None) -> str:
    """
    Đọc file knowledge base và trả về nội dung dạng chuỗi.

    Args:
        path: đường dẫn tới file text.
              Mặc định: data/knowledge_base.txt (thư mục gốc của project)

    Returns:
        Nội dung file dưới dạng str
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "knowledge_base.txt"
    return Path(path).read_text(encoding="utf-8")


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Chia văn bản thành các đoạn nhỏ (chunks) để index.

    Dùng RecursiveCharacterTextSplitter — tách ưu tiên theo đoạn văn, câu, rồi ký tự.

    Args:
        text         : văn bản cần chia
        chunk_size   : số ký tự tối đa mỗi chunk (mặc định: 500)
        chunk_overlap: số ký tự chồng lên nhau giữa 2 chunks liên tiếp (mặc định: 50)

    Returns:
        list[str] — danh sách các chuỗi chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def build_vectorstore(chunks: list, embeddings):
    """
    Tạo FAISS vectorstore từ danh sách chunks và embeddings.

    Args:
        chunks    : list[str] — danh sách text chunks đã chia
        embeddings: Embeddings instance (từ get_embeddings())

    Returns:
        FAISS vectorstore đã được index và sẵn sàng dùng để retrieve
    """
    from langchain_community.vectorstores import FAISS

    model_name = str(getattr(embeddings, "model", ""))
    content_digest = hashlib.sha256("\0".join(chunks).encode("utf-8")).hexdigest()
    cache_key = (type(embeddings).__module__, type(embeddings).__qualname__, model_name, content_digest)
    if cache_key in _VECTORSTORE_CACHE:
        print(f"♻️  Tái sử dụng FAISS index {len(chunks)} chunks trong phiên hiện tại.")
        return _VECTORSTORE_CACHE[cache_key]

    print(f"🔨 Đang tạo FAISS index từ {len(chunks)} chunks ...")
    vectors = embed_documents_in_batches(chunks, embeddings)
    vectorstore = FAISS.from_embeddings(zip(chunks, vectors), embeddings)
    _VECTORSTORE_CACHE[cache_key] = vectorstore
    print("✅ FAISS vectorstore đã sẵn sàng.")
    return vectorstore
