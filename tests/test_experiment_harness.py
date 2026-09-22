import json
import tempfile
import unittest
from pathlib import Path

from src.prompt_optimizer import optimize_prompt, parse_optimizers
from src.prompts import (
    COT_PROMPT,
    EXPERT_PERSONA_PROMPT,
    SIMPLEQA_FINAL_ANSWER_FORMAT,
    SIMPLEQA_QUESTION_PROMPT,
    SIMPLEQA_TASK_PROMPT,
    UNCERTAINTY_PROMPT,
)
from src.task_runner import TaskRunner
from src.utils import TASK_MAPPING


class FakeLLM:
    model_id = "fake-model"

    def call_llm(self, prompt, max_tokens):
        return "=== FINAL ANSWER ===\n4\n=== END FINAL ANSWER ==="

    def get_model_info(self):
        return {"provider": "FakeLLM"}


class TestExperimentHarness(unittest.TestCase):
    def test_prompt_composition_preserves_the_defined_order(self):
        prompt = optimize_prompt(
            "What is 2 + 2?",
            parse_optimizers("uncertainty,cot,expert_persona"),
            TASK_MAPPING["simpleqa"],
        )

        ordered_parts = (
            EXPERT_PERSONA_PROMPT,
            SIMPLEQA_TASK_PROMPT,
            UNCERTAINTY_PROMPT,
            COT_PROMPT,
            SIMPLEQA_FINAL_ANSWER_FORMAT,
            SIMPLEQA_QUESTION_PROMPT.format(question="What is 2 + 2?"),
        )
        positions = [prompt.index(part) for part in ordered_parts]
        self.assertEqual(positions, sorted(positions))

    def test_runner_writes_results_under_output_path(self):
        with tempfile.TemporaryDirectory() as output_path:
            runner = TaskRunner(
                llm=FakeLLM(),
                task="simpleqa_small",
                questions=["What is 2 + 2?"],
                optimizers="none",
                output_path=output_path,
            )

            runner.run_experiments()

            result_path = (
                Path(output_path)
                / "result"
                / "fake-model_simpleqa_small_none_results.json"
            )
            self.assertTrue(result_path.is_file())
            records = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(records[0]["Final Answer Section"], "4")


if __name__ == "__main__":
    unittest.main()
