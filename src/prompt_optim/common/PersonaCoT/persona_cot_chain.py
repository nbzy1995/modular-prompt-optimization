import os, json

_SYS = (
    "You are a highly knowledgeable and careful scientific expert. "
    "Use only reliable facts. Think step by step. "
    "After your reasoning, provide one line starting with 'Final answer:' "
    "containing only the final answer."
)

_TEMPLATE = """{sys}

Q: {q}
A:"""

def _postprocess(text: str) -> str:
    t = (text or "").strip()
    marker = "final answer:"
    i = t.lower().rfind(marker)
    if i != -1:
        return t[i+len(marker):].strip()
    return t

class PersonaCoTChain:
    def __init__(self, llm, task, questions, model_id: str):
        self.llm = llm
        self.task = task
        self.questions = questions
        self.model_id = model_id
        os.makedirs("result", exist_ok=True)
        self.out_path = f"result/{self.model_id}_{self.task}_persona_cot_results.json"

    def _extract(self, item, i):
        if isinstance(item, dict):
            return item.get("id", f"q{i+1}"), item.get("question",""), item.get("answer","")
        return f"q{i+1}", str(item), ""

    def build_prompt(self, q: str) -> str:
        return _TEMPLATE.format(sys=_SYS, q=q)

    def run(self, max_tokens: int = 768):
        rows = []
        for i, it in enumerate(self.questions):
            qid, qtext, gold = self._extract(it, i)
            prompt = self.build_prompt(qtext)
            raw = self.llm.call_llm(prompt, max_tokens)
            final = _postprocess(raw)
            rows.append({
                "id": qid, "Question": qtext, "Gold": gold,
                "Final Answer": final, "Final Refined Answer": final,
                "Prompt": prompt, "Raw": (raw or "").strip()
            })
            print(f"[{i+1}/{len(self.questions)}] {qtext[:60]}... -> {final[:120]}")
            # Optionally print 'Raw' to inspect reasoning:
            # print(raw)

        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved Persona+CoT results to {self.out_path}")
