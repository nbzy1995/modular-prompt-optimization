from typing import Dict

class ZeroShotCOTPipeline:
    """
    Faithful two-stage Zero-shot-CoT (Kojima et al., 2022):
      1) Reasoning extraction with trigger
      2) Answer extraction with format-specific trigger
    """
    def __init__(self, model_client, task: str):
        self.client = model_client
        self.task = task
        self.reason_trigger = "Let's think step by step."
        self.answer_triggers = {
            "multi_arith": "Therefore, the answer (arabic numerals) is",
            "commonsenseqa": "Therefore, among A through E, the answer is",
        }

    def build_question_text(self, sample: Dict) -> str:
        if self.task == "commonsenseqa":
            choices = sample["choices"]
            ordered = [f"({lab}) {choices[lab]}" for lab in ["A","B","C","D","E"] if lab in choices]
            choices_str = " Answer Choices: " + " ".join(ordered)
            return f"Q: {sample['question'].strip()}{choices_str}"
        return f"Q: {sample['question'].strip()}"

    def run_sample(self, sample: Dict) -> Dict:
        q_text = self.build_question_text(sample)

        # Stage 1 — reasoning extraction
        p1 = f"{q_text}\nA: {self.reason_trigger}"
        rationale = self.client.generate(p1, max_tokens=256)

        # Stage 2 — answer extraction
        ans_trigger = self.answer_triggers[self.task]
        p2 = f"{q_text}\nA: {self.reason_trigger} {rationale.strip()}\n{ans_trigger}"
        final = self.client.generate(p2, max_tokens=64)

        return {
            "question": sample["question"],
            "ground_truth": str(sample.get("answer", "")).strip(),
            "reasoning": rationale.strip(),
            "final_prompt": p2,
            "raw_answer": final.strip(),
        }


class DirectAnswerPipeline:
    """
    Standard zero-shot baseline with format-specific answer prompts.
    No chain-of-thought.
    """
    def __init__(self, model_client, task: str):
        self.client = model_client
        self.task = task
        self.answer_triggers = {
            "multi_arith": "The answer (arabic numerals) is",
            "commonsenseqa": "Among A through E, the answer is",
        }

    def build_question_text(self, sample: Dict) -> str:
        if self.task == "commonsenseqa":
            choices = sample["choices"]
            ordered = [f"({lab}) {choices[lab]}" for lab in ["A","B","C","D","E"] if lab in choices]
            choices_str = " Answer Choices: " + " ".join(ordered)
            return f"Q: {sample['question'].strip()}{choices_str}"
        return f"Q: {sample['question'].strip()}"

    def run_sample(self, sample: Dict) -> Dict:
        q_text = self.build_question_text(sample)
        ans_trigger = self.answer_triggers[self.task]
        prompt = f"{q_text}\nA: {ans_trigger}"
        response = self.client.generate(prompt, max_tokens=64).strip()

        return {
            "question": sample["question"],
            "ground_truth": str(sample.get("answer", "")).strip(),
            "reasoning": "",
            "raw_answer": response,
        }
