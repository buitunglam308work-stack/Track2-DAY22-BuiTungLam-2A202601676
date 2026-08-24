"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import argparse
import hashlib
import sys
import json
import math
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.console import configure_utf8_console

configure_utf8_console()

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from utils.retry import invoke_with_retry
from qa_pairs import QA_PAIRS


DATA_DIR = Path(__file__).parent.parent / "data"
EVIDENCE_DIR = Path(__file__).parent.parent / "evidence"
RETRIEVAL_CONFIG = {"chunk_size": 500, "chunk_overlap": 50, "k": 3}
RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")
RAGAS_RUN_CONFIG = RunConfig(
    timeout=180,
    max_retries=3,
    max_wait=10,
    max_workers=4,
    log_tenacity=False,
)


def _checkpoint_path(prompt_version: str) -> Path:
    return DATA_DIR / f"ragas_checkpoint_{prompt_version}.json"


def load_checkpoint(path: Path, expected_fingerprint: str = None) -> list:
    """Load completed raw samples, returning an empty list if none exists."""
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        # Backwards compatible for callers/tests. A fingerprinted run must not
        # silently reuse an older, unverifiable prompt/model checkpoint.
        return payload if expected_fingerprint is None else []
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(f"Checkpoint must contain results: {path}")
    if expected_fingerprint and payload.get("fingerprint") != expected_fingerprint:
        return []
    return payload["results"]


def save_checkpoint(path: Path, results: list, fingerprint: str = None) -> None:
    """Atomically persist raw outputs so an interrupted run can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {"fingerprint": fingerprint, "results": results} if fingerprint else results
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_report(report: dict) -> None:
    """Validate the submission report shape before writing/copying it."""
    required = {"prompt_v1_scores", "prompt_v2_scores", "target_met", "sample_counts", "model", "retrieval_config", "winner", "analysis"}
    missing = required.difference(report)
    if missing:
        raise ValueError(f"RAGAS report missing fields: {sorted(missing)}")
    metric_names = {"faithfulness", "answer_relevancy", "context_recall", "context_precision"}
    for version in ("prompt_v1_scores", "prompt_v2_scores"):
        if set(report[version]) != metric_names:
            raise ValueError(f"RAGAS report metrics invalid for {version}")
        if any(not np.isfinite(float(report[version][metric])) for metric in metric_names):
            raise ValueError(f"RAGAS report has a non-finite metric in {version}")
    if report["sample_counts"] != {"v1": 50, "v2": 50}:
        raise ValueError("RAGAS report must contain 50 samples for each version")


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
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

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}
PROMPT_SYSTEMS = {"v1": SYSTEM_V1, "v2": SYSTEM_V2}


def selected_model_metadata() -> dict:
    """Return the exact provider/model pair responsible for checkpoint outputs."""
    model_fields = {
        "llm": {
            "openai": "OPENAI_MODEL", "gemini": "GEMINI_MODEL",
            "anthropic": "ANTHROPIC_MODEL", "ollama": "OLLAMA_MODEL",
            "openrouter": "OPENROUTER_MODEL",
        },
        "embedding": {
            "openai": "OPENAI_EMBEDDING_MODEL", "gemini": "GEMINI_EMBEDDING_MODEL",
            "ollama": "OLLAMA_EMBEDDING_MODEL",
        },
    }
    selected = {"provider": config.PROVIDER}
    for kind, fields in model_fields.items():
        selected[kind] = getattr(config, fields.get(config.PROVIDER, fields["openai"]), "unknown")
    return selected


def prompt_fingerprint(prompt_version: str) -> str:
    """Bind a checkpoint to prompt text, models, and retrieval configuration."""
    payload = {
        "prompt_version": prompt_version,
        "system_prompt": PROMPT_SYSTEMS[prompt_version],
        "model": selected_model_metadata(),
        "retrieval": RETRIEVAL_CONFIG,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    docs = invoke_with_retry(lambda: retriever.invoke(question))

    contexts = [doc.page_content for doc in docs]

    ctx_str = "\n\n".join(contexts)

    chain = prompt | llm | StrOutputParser()
    answer = invoke_with_retry(lambda: chain.invoke({
        "context": ctx_str,
        "question": question,
    }))

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str, checkpoint_path: Path = None) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    checkpoint_path = checkpoint_path or _checkpoint_path(prompt_version)
    fingerprint = prompt_fingerprint(prompt_version)
    results = load_checkpoint(checkpoint_path, expected_fingerprint=fingerprint)
    if len(results) > len(QA_PAIRS):
        raise ValueError(f"Checkpoint has too many samples: {checkpoint_path}")
    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        if i <= len(results) and results[i - 1].get("question") == qa["question"]:
            print(f"  [{i:02d}/50] resumed from checkpoint")
            continue
        if i <= len(results):
            results = results[: i - 1]
        out = run_rag(retriever, llm, prompt, qa["question"])

        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        save_checkpoint(checkpoint_path, results, fingerprint=fingerprint)
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    return EvaluationDataset(samples=samples)


def validate_metric_results(result, expected_count: int) -> dict:
    """Return means only when every metric has a complete finite result column."""
    if expected_count <= 0:
        raise ValueError("RAGAS evaluation requires at least one sample")

    scores = {}
    for key in RAGAS_METRICS:
        try:
            raw = result[key]
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError(f"RAGAS result missing metric: {key}") from exc
        values = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        if len(values) != expected_count:
            raise ValueError(
                f"RAGAS metric {key} returned {len(values)} samples; expected {expected_count}"
            )
        numeric = []
        for index, value in enumerate(values):
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"RAGAS metric {key} has non-numeric sample {index}") from exc
            if not np.isfinite(number):
                raise ValueError(f"RAGAS metric {key} has invalid sample {index}: {value!r}")
            numeric.append(number)
        scores[key] = float(np.mean(numeric))
    return scores


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    dataset = build_ragas_dataset(rag_results)

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator
    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()

    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    result = invoke_with_retry(
        lambda: evaluate(
            dataset,
            metrics=metrics,
            llm=llm_eval,
            embeddings=emb_eval,
            run_config=RAGAS_RUN_CONFIG,
            batch_size=4,
            raise_exceptions=True,
        ),
        attempts=3,
        base_delay=2.0,
    )

    scores = validate_metric_results(result, expected_count=len(rag_results))

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the complete RAGAS comparison.")
    parser.add_argument(
        "--only-v2",
        action="store_true",
        help="Regenerate/evaluate V2 while reusing the validated V1 baseline report.",
    )
    args = parser.parse_args(argv)

    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore = setup_vectorstore()

    if args.only_v2:
        existing_path = DATA_DIR / "ragas_report.json"
        if not existing_path.exists():
            raise FileNotFoundError("--only-v2 requires an existing validated RAGAS report")
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        validate_report(existing)
        if existing["model"] != selected_model_metadata():
            raise ValueError("--only-v2 cannot reuse V1 scores from a different provider/model")
        if existing["retrieval_config"] != RETRIEVAL_CONFIG:
            raise ValueError("--only-v2 cannot reuse V1 scores with different retrieval settings")
        v1_results = load_checkpoint(_checkpoint_path("v1"))
        if len(v1_results) != len(QA_PAIRS) or any(
            row.get("question") != qa["question"] for row, qa in zip(v1_results, QA_PAIRS)
        ):
            raise ValueError("--only-v2 requires the complete matching V1 raw checkpoint")
        v1_scores = existing["prompt_v1_scores"]
        print("\n♻️  Reusing the validated 50-sample V1 baseline scores; evaluating only V2.")
    else:
        v1_results = collect_rag_outputs(vectorstore, "v1")
        v1_scores = run_ragas_eval(v1_results, "v1")

    v2_results = collect_rag_outputs(vectorstore, "v2")
    v2_scores = run_ragas_eval(v2_results, "v2")

    metric_names = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

    def metric_winner(metric: str) -> str:
        """Return a stable winner label, treating negligible float noise as a tie."""
        score_v1, score_v2 = v1_scores[metric], v2_scores[metric]
        if math.isclose(score_v1, score_v2, rel_tol=0.0, abs_tol=1e-6):
            return "tie"
        return "v1" if score_v1 > score_v2 else "v2"

    winner_by_metric = {metric: metric_winner(metric) for metric in metric_names}

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    winner_labels = {"v1": "← V1", "v2": "← V2", "tie": "= tie"}
    for metric in metric_names:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner = winner_labels[winner_by_metric[metric]]
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    faithfulness_winner = winner_by_metric["faithfulness"]
    faithfulness_delta = abs(v1_scores["faithfulness"] - v2_scores["faithfulness"])
    relevancy_delta = abs(v1_scores["answer_relevancy"] - v2_scores["answer_relevancy"])
    if faithfulness_winner == "tie":
        analysis = (
            "V1 và V2 có faithfulness bằng nhau; cần ưu tiên độ ngắn gọn hoặc "
            "cấu trúc theo mục tiêu sử dụng."
        )
    else:
        style_reason = (
            "định dạng trực tiếp 2-4 câu của V1 giảm các chi tiết thừa"
            if faithfulness_winner == "v1"
            else "cấu trúc kết luận kèm sự thật hỗ trợ, không có meta-claim của V2"
        )
        analysis = (
            f"{faithfulness_winner.upper()} có faithfulness cao hơn {faithfulness_delta:.4f}; "
            f"chênh lệch answer relevancy tuyệt đối là {relevancy_delta:.4f}. "
            "Hai phiên bản dùng cùng retriever và context, nên khác biệt chủ yếu đến "
            f"từ cách định dạng câu trả lời: {style_reason}."
        )
    selected_model = selected_model_metadata()
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
        "sample_counts": {"v1": len(v1_results), "v2": len(v2_results)},
        "model": selected_model,
        "retrieval_config": RETRIEVAL_CONFIG,
        "winner": {"by_metric": winner_by_metric, "faithfulness": faithfulness_winner},
        "analysis": analysis,
        "checkpoints": {"v1": "data/ragas_checkpoint_v1.json", "v2": "data/ragas_checkpoint_v2.json"},
        "prompt_fingerprints": {
            "v1": prompt_fingerprint("v1"),
            "v2": prompt_fingerprint("v2"),
        },
    }
    validate_report(report)
    report_path = DATA_DIR / "ragas_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "03_ragas_report.json").write_text(
        report_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main()
