import os, json

_SYS = (
    "You are a highly knowledgeable and careful scientific expert. "
    "Use only reliable facts. "
    "If you are not certain, reply exactly: I don't know"
)

_TEMPLATE = """{sys}

Q: {q}
A:"""

class CombinedChain:
    def __init__(self, llm, task, questions, model_id: str):
        self.llm = llm
        self.task = task
        self.questions = questions
        self.model_id = model_id
        os.makedirs("result", exist_ok=True)
        self.out_path = f"result/{self.model_id}_{self.task}_combined_results.json"

    def _extract(self, item, i):
        if isinstance(item, dict):
            return item.get("id", f"q{i+1}"), item.get("question",""), item.get("answer","")
        return f"q{i+1}", str(item), ""

    def build_prompt(self, q: str) -> str:
        return _TEMPLATE.format(sys=_SYS, q=q)

    def run(self, max_tokens: int = 512):
        rows = []
        for i, it in enumerate(self.questions):
            qid, qtext, gold = self._extract(it, i)
            prompt = self.build_prompt(qtext)
            raw = self.llm.call_llm(prompt, max_tokens)
            ans = (raw or "").strip()
            rows.append({
                "id": qid, "Question": qtext, "Gold": gold,
                "Final Answer": ans, "Final Refined Answer": ans, "Prompt": prompt
            })
            print(f"[{i+1}/{len(self.questions)}] {qtext[:60]}... -> {ans[:120]}")
        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved Combined results to {self.out_path}")
