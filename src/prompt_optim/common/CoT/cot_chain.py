import os, json

_SYS = (
    "Think step by step to reach a correct answer. "
    "After your reasoning, provide a single line starting with 'Final answer:' "
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

class ChainOfThought:
    def __init__(self, llm, task, questions, model_id: str):
        self.llm = llm
        self.task = task
        self.questions = questions
        self.model_id = model_id
        os.makedirs("result", exist_ok=True)
        self.out_path = f"result/{self.model_id}_{self.task}_cot_results.json"

    def _extract(self, item, i):
        if isinstance(item, dict):
            return item.get("id", f"q{i+1}"), item.get("question",""), item.get("answer","")
        return f"q{i+1}", str(item), ""

    def build_prompt(self, q: str) -> str:
        return _TEMPLATE.format(sys=_SYS, q=q)

    def run(self, max_tokens: int = 768):
        tech = "cot_uncertainty"   # ← set this per file: persona, uncertainty, combined, cot, persona_cot, cot_uncertainty

        # --- setup paths & resume ---
        import os, json, time
        os.makedirs("result", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

        result_path = f"result/{self.model_id}_{self.task}_{tech}_results.json"
        ckpt_path = f"checkpoints/{self.model_id}_{self.task}_{tech}_checkpoint.json"

        # load previous results if resuming
        results = []
        start_idx = 0
        if os.path.exists(ckpt_path):
            try:
                with open(ckpt_path, "r", encoding="utf-8") as f:
                    ck = json.load(f)
                    start_idx = int(ck.get("last_completed_index", -1)) + 1
                    prev = ck.get("completed_results", [])
                    if isinstance(prev, list):
                        results = prev
                print(f"🔄 Resuming from question {start_idx+1}/{len(self.questions)}")
                print(f"   Checkpoint: {ckpt_path}")
            except Exception:
                pass
        else:
            print(f"🆕 Starting fresh experiment with {len(self.questions)} questions")

        # helper: extract question/gold
        def _extract(item, i):
            if isinstance(item, dict):
                return item.get("id", f"q{i+1}"), item.get("question", ""), item.get("answer", "")
            return f"q{i+1}", str(item), ""

        # helper: safe 60-char preview with ellipsis
        def _preview(s, n=60):
            s = str(s)
            return (s[:n-3] + "...") if len(s) > n else s

        # provider name for pretty line
        try:
            info = self.llm.get_model_info()
            provider = (info or {}).get("provider", "LLM")
        except Exception:
            provider = "LLM"

        total = len(self.questions)

        for i in range(start_idx, total):
            qid, qtext, gold = _extract(self.questions[i], i)

            # ---- progress header (your desired format) ----
            pct = (i+1) / total * 100.0
            print(f"\n📊 Progress: {i+1}/{total} ({pct:.1f}%)")
            print(f"🔄 Current question: {_preview(qtext)}")
            print(f"🤖 Using: {provider} ({self.model_id})")

            # ---- build and call (each chain’s build_prompt stays as you wrote) ----
            prompt = self.build_prompt(qtext)
            raw = self.llm.call_llm(prompt, max_tokens)
            final = raw if isinstance(raw, str) else str(raw)

            # optional chain-specific postprocess:
            if hasattr(self, "_postprocess"):
                try:
                    final = self._postprocess(final)
                except Exception:
                    pass

            print(f"Answer: {final.splitlines()[0][:300]}")
            print("----------------------")

            # ---- save to in-memory results ----
            results.append({
                "id": qid,
                "Question": qtext,
                "Gold": gold,
                "Prompt": prompt,
                # keep both keys so existing evaluators work:
                "Final Answer": final,
                "Final Refined Answer": final,
            })

            # ---- checkpoint after every question ----
            ck = {
                "model_id": self.model_id,
                "task": self.task,
                "technique": tech,
                "last_completed_index": i,
                "total_questions": total,
                "completed_results": results,
                "timestamp": time.time(),
            }
            try:
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    json.dump(ck, f, ensure_ascii=False, indent=2)
                print(f"✅ Checkpoint saved ({i+1}/{total} completed)")
            except Exception as e:
                print(f"⚠️ Failed to save checkpoint: {e}")

        # ---- write final results JSON ----
        try:
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n🎉 Done. Results saved to: {result_path}")
            # optional: remove checkpoint here if you want a clean end
            # os.remove(ckpt_path)
        except Exception as e:
            print(f"⚠️ Failed to save results: {e}")

