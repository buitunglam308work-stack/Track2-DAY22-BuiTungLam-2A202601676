"""Offline smoke tests for the deterministic and pure lab components."""

import ast
import importlib
import importlib.util
import json
import tempfile
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from utils.console import configure_utf8_console  # noqa: E402

configure_utf8_console()


class SourceCompletenessTests(unittest.TestCase):
    def test_no_scaffold_ellipsis_in_executable_source(self):
        for path in SRC.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            ellipses = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value is Ellipsis]
            self.assertFalse(ellipses, f"unimplemented Ellipsis remains in {path.name}")

    def test_qa_dataset_has_fifty_pairs(self):
        qa = importlib.import_module("qa_pairs")
        self.assertEqual(len(qa.SAMPLE_QUESTIONS), 50)
        self.assertEqual(len(qa.QA_PAIRS), 50)


class DataLoaderTests(unittest.TestCase):
    def test_embedding_batches_and_retry_hint(self):
        loader = importlib.import_module("utils.data_loader")

        class FakeEmbeddings:
            def __init__(self):
                self.calls = []
                self.failed = False

            def embed_documents(self, texts):
                self.calls.append(len(texts))
                if not self.failed:
                    self.failed = True
                    raise RuntimeError("Please retry in 3.5s")
                return [[float(i), 1.0] for i, _ in enumerate(texts)]

        fake = FakeEmbeddings()
        delays = []
        vectors = loader.embed_documents_in_batches(
            [f"chunk-{i}" for i in range(105)], fake, batch_size=50, sleep=delays.append
        )
        self.assertEqual(len(vectors), 105)
        self.assertEqual(fake.calls, [50, 50, 50, 5])
        self.assertEqual(delays, [4.5])


class RoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module("02_prompt_hub_ab_routing")

    def test_routing_is_repeatable_and_balanced(self):
        get_version = self.module.get_prompt_version
        ids = [f"req-{i:04d}" for i in range(50)]
        first = [get_version(value) for value in ids]
        second = [get_version(value) for value in ids]
        self.assertEqual(first, second)
        self.assertIn(self.module.PROMPT_V1_NAME, first)
        self.assertIn(self.module.PROMPT_V2_NAME, first)

    def test_prompts_accept_context(self):
        for prompt in (self.module.PROMPT_V1, self.module.PROMPT_V2):
            self.assertIn("context", prompt.input_variables)
            self.assertIn("question", prompt.input_variables)


class RunAllTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module("run_all")

    def test_step4_isolates_parent_step_argument(self):
        received = []
        fake_module = SimpleNamespace(main=lambda argv=None: received.append(argv))
        with mock.patch.object(self.module.importlib, "import_module", return_value=fake_module):
            self.assertTrue(self.module.run_step(4))
        self.assertEqual(received, [[]])

    def test_main_returns_nonzero_when_selected_step_fails(self):
        with mock.patch.object(self.module, "run_step", return_value=False) as run_step:
            with mock.patch.object(sys, "argv", ["run_all.py", "--step", "4"]):
                self.assertEqual(self.module.main(), 1)
        run_step.assert_called_once_with(4)


class ConfigTests(unittest.TestCase):
    def test_langsmith_client_includes_optional_workspace(self):
        config = importlib.import_module("config")
        fake_client = mock.Mock()
        with mock.patch("langsmith.Client", return_value=fake_client) as client:
            with mock.patch.object(config, "LANGSMITH_WORKSPACE_ID", "workspace-id"):
                self.assertIs(config.get_langsmith_client(), fake_client)
        self.assertEqual(client.call_args.kwargs["workspace_id"], "workspace-id")
        self.assertEqual(client.call_args.kwargs["api_key"], config.LANGSMITH_API_KEY)


class LLMFactoryTests(unittest.TestCase):
    def test_gemini_uses_shared_rate_limiter_without_fixed_temperature_warning(self):
        factory = importlib.import_module("utils.llm_factory")
        with mock.patch("langchain_google_genai.ChatGoogleGenerativeAI") as chat:
            with mock.patch.object(factory.config, "GEMINI_MODEL", "gemini-3.6-flash"):
                factory.get_llm("gemini", temperature=0)
        kwargs = chat.call_args.kwargs
        self.assertIn("rate_limiter", kwargs)
        self.assertNotIn("temperature", kwargs)


class RetryAndCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.module = importlib.import_module("03_ragas_evaluation")
        except Exception as exc:
            cls.module = None
            cls.import_error = exc

    def test_retry_uses_exponential_delays(self):
        if self.module is None:
            self.skipTest(f"RAGAS stack unavailable: {self.import_error}")
        calls = {"count": 0}
        delays = []

        def flaky():
            calls["count"] += 1
            if calls["count"] < 3:
                raise RuntimeError("transient")
            return "ok"

        value = self.module.invoke_with_retry(
            flaky, attempts=4, base_delay=0.25, sleep=delays.append, label="test"
        )
        self.assertEqual(value, "ok")
        self.assertEqual(delays, [0.25, 0.5])

    def test_checkpoint_round_trip_and_report_schema(self):
        if self.module is None:
            self.skipTest(f"RAGAS stack unavailable: {self.import_error}")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            rows = [{"question": "q", "answer": "a", "contexts": ["c"], "reference": "r"}]
            self.module.save_checkpoint(path, rows)
            self.assertEqual(self.module.load_checkpoint(path), rows)
            self.assertEqual(self.module.load_checkpoint(path, expected_fingerprint="new"), [])
            self.module.save_checkpoint(path, rows, fingerprint="current")
            self.assertEqual(
                self.module.load_checkpoint(path, expected_fingerprint="current"), rows
            )
            self.assertEqual(self.module.load_checkpoint(path, expected_fingerprint="stale"), [])
        scores = {name: 0.9 for name in ("faithfulness", "answer_relevancy", "context_recall", "context_precision")}
        report = {
            "prompt_v1_scores": scores,
            "prompt_v2_scores": scores,
            "target_met": True,
            "sample_counts": {"v1": 50, "v2": 50},
            "model": {}, "retrieval_config": {}, "winner": {}, "analysis": "ok",
        }
        self.module.validate_report(report)

    def test_step2_step3_prompt_parity(self):
        if self.module is None:
            self.skipTest(f"RAGAS stack unavailable: {self.import_error}")
        step2 = importlib.import_module("02_prompt_hub_ab_routing")
        self.assertEqual(step2.SYSTEM_V1, self.module.SYSTEM_V1)
        self.assertEqual(step2.SYSTEM_V2, self.module.SYSTEM_V2)

    def test_metric_validation_rejects_missing_or_invalid_samples(self):
        if self.module is None:
            self.skipTest(f"RAGAS stack unavailable: {self.import_error}")
        complete = {name: [0.8, 0.9] for name in self.module.RAGAS_METRICS}
        scores = self.module.validate_metric_results(complete, expected_count=2)
        self.assertAlmostEqual(scores["faithfulness"], 0.85)

        incomplete = dict(complete)
        incomplete["context_recall"] = [0.8]
        with self.assertRaises(ValueError):
            self.module.validate_metric_results(incomplete, expected_count=2)

        invalid = dict(complete)
        invalid["faithfulness"] = [0.8, float("nan")]
        with self.assertRaises(ValueError):
            self.module.validate_metric_results(invalid, expected_count=2)

    def test_ragas_evaluate_uses_bounded_strict_options(self):
        if self.module is None:
            self.skipTest(f"RAGAS stack unavailable: {self.import_error}")
        fake_result = {name: [0.8, 0.9] for name in self.module.RAGAS_METRICS}
        rows = [
            {"question": "q1", "answer": "a1", "contexts": ["c1"], "reference": "r1"},
            {"question": "q2", "answer": "a2", "contexts": ["c2"], "reference": "r2"},
        ]
        with mock.patch.object(self.module, "evaluate", return_value=fake_result) as evaluate:
            with mock.patch.object(self.module, "get_llm", return_value=mock.Mock()):
                with mock.patch.object(self.module, "get_embeddings", return_value=mock.Mock()):
                    scores = self.module.run_ragas_eval(rows, "v1")
        self.assertAlmostEqual(scores["faithfulness"], 0.85)
        kwargs = evaluate.call_args.kwargs
        self.assertTrue(kwargs["raise_exceptions"])
        self.assertEqual(kwargs["batch_size"], 4)
        self.assertIs(kwargs["run_config"], self.module.RAGAS_RUN_CONFIG)


@unittest.skipUnless(importlib.util.find_spec("guardrails"), "guardrails-ai is not installed")
class GuardrailsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module("04_guardrails_validator")
        cls.on_fail = cls.module.OnFailAction.FIX

    def test_pii_detector_redacts_all_supported_types(self):
        guard = self.module.Guard().use(self.module.PIIDetector(on_fail=self.on_fail))
        text = "email a@example.com phone 555-123-4567 ssn 123-45-6789 card 4111 1111 1111 1111"
        result = guard.validate(text)
        self.assertNotIn("a@example.com", result.validated_output)
        self.assertNotIn("555-123-4567", result.validated_output)
        self.assertNotIn("123-45-6789", result.validated_output)
        self.assertNotIn("4111 1111 1111 1111", result.validated_output)
        self.assertIn("EMAIL_REDACTED", result.validated_output)
        self.assertIn("PHONE_REDACTED", result.validated_output)
        self.assertIn("SSN_REDACTED", result.validated_output)
        self.assertIn("CREDIT_CARD_REDACTED", result.validated_output)

    def test_json_formatter_repairs_and_falls_back(self):
        guard = self.module.Guard().use(self.module.JSONFormatter(on_fail=self.on_fail))
        repaired = [
            '```json\n{"name": "Bob"}\n```',
            "{'name': 'Charlie', 'score': 95}",
            '{"items": ["a",],}',
        ]
        for text in repaired:
            output = guard.validate(text).validated_output
            self.assertIsInstance(json.loads(output), dict)

        fallback = guard.validate("not json {]").validated_output
        payload = json.loads(fallback)
        self.assertEqual(payload["error"], "Không thể phân tích JSON")

    def test_demo_argparse_accepts_each_mode(self):
        # The parser is exercised without asserting console text.
        self.module.main(["--demo", "pii"])
        self.module.main(["--demo", "json"])


if __name__ == "__main__":
    unittest.main()
