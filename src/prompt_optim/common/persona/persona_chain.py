# src/prompt_optim/common/persona_chain.py
import os, json

_PERSONA = (
    "You are a highly knowledgeable and careful scientific expert. "
    "Answer using only reliable, well-established facts. "
    "Be concise and precise."
)

_TEMPLATE = """{sys}

Q: {q}
A:"""

class PersonaChain:
    def __init__(self, llm, task, questions, model_id: str):
        self.llm = llm
        self.task = task
        self.questions = questions
        self.model_id = model_id
        self.results = []
        os.makedirs("result", exist_ok=True)

    def build_prompt(self, q: str) -> str:
        return _TEMPLATE.format(sys=_PERSONA, q=q)

    def run(self):
        for i, q in enumerate(self.questions):
            prompt = self.build_prompt(q)
            answer = self.llm.call_llm(prompt, max_tokens=512)
            result = {"Question": q, "Answer": answer}
            self.results.append(result)
            print(f"[{i+1}/{len(self.questions)}] {q[:60]}... -> {answer}")

        out_path = f"result/{self.model_id}_{self.task}_persona_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        print(f"✅ Saved Persona results to {out_path}")
