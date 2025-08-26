import os
import json
import time
import sys
from typing import Dict, List, Any


class ChainOfVerification:
    """Chain of Verification that uses simple LLM interface."""
    
    def __init__(self, llm, task: str, questions: List[str]):
        # Store LLM and configuration
        self.llm = llm
        self.model_id = llm.model_id
        self.task = task
        self.setting = "joint"  # Always use joint setting
        self.questions = questions
        
        # Get task config (use wikidata config for test task)
        from ...utils import TASK_MAPPING
        actual_task = "wikidata" if task == "test" else task
        self.task_config = TASK_MAPPING.get(actual_task, None)
        if self.task_config is None:
            print(f"Invalid task. Valid tasks are: {', '.join(TASK_MAPPING.keys())}")
            sys.exit()
        
        # Checkpoint setup - use current working directory  
        self.checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir,
            f"{self.model_id}_{self.task}_{self.setting}_checkpoint.json"
        )
        
        # Load checkpoint if exists
        self.checkpoint_data = self.load_checkpoint()
        self.start_question_index = self.checkpoint_data.get("last_completed_index", -1) + 1
        
        if self.start_question_index > 0:
            print(f"🔄 Resuming from question {self.start_question_index + 1}/{len(questions)}")
            print(f"   Checkpoint: {self.checkpoint_file}")
        else:
            print(f"🆕 Starting fresh experiment with {len(questions)} questions")

    def load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint data if exists."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading checkpoint: {e}")
                return {}
        return {}

    def save_checkpoint(self, question_index: int, results: List[Dict[str, str]]):
        """Save current progress to checkpoint file."""
        try:
            checkpoint_data = {
                "model_id": self.model_id,
                "task": self.task,
                "setting": self.setting,
                "last_completed_index": question_index,
                "total_questions": len(self.questions),
                "completed_results": results,
                "timestamp": time.time()
            }
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Checkpoint saved ({question_index + 1}/{len(self.questions)} completed)")
        except Exception as e:
            print(f"⚠️ Error saving checkpoint: {e}")

    def cleanup_checkpoint(self):
        """Remove checkpoint file after successful completion."""
        try:
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
                print("🧹 Checkpoint cleaned up")
        except Exception as e:
            print(f"⚠️ Error cleaning up checkpoint: {e}")

    def call_llm(self, prompt: str, max_tokens: int) -> str:
        """Call LLM using the simple interface."""
        return self.llm.call_llm(prompt, max_tokens)

    def process_prompt(self, prompt: str, command: str = "") -> str:
        """Process prompt (no special formatting needed)."""
        return prompt

    def get_baseline_response(self, question: str) -> str:
        """Get baseline response for a question."""
        baseline_prompt = self.task_config.baseline_prompt.format(
            original_question=question
        )
        processed_prompt = self.process_prompt(
            baseline_prompt, 
            self.task_config.baseline_command
        )
        return self.call_llm(processed_prompt, self.task_config.max_tokens)

    def run_joint_chain(self, question: str, baseline_response: str):
        """Run joint CoVe chain."""
        # Create and Execute Plan
        plan_and_execution_prompt = self.task_config.joint.plan_and_execute_prompt.format(
            original_question=question,
            baseline_response=baseline_response,
        )

        plan_and_execution_response = self.call_llm(
            self.process_prompt(plan_and_execution_prompt, self.task_config.joint.plan_and_execute_command),
            self.task_config.joint.max_tokens_plan_and_execute
        )

        # Verify
        verify_prompt = self.task_config.joint.verify_prompt.format(
            original_question=question,
            baseline_response=baseline_response,
            verification_questions_and_answers=plan_and_execution_response,
        )

        verify_response = self.call_llm(
            self.process_prompt(verify_prompt, self.task_config.joint.verify_command),
            self.task_config.joint.max_tokens_verify
        )

        return plan_and_execution_response, verify_response

    def print_result(self, result: Dict[str, str]):
        """Print result."""
        for key, value in result.items():
            print(f"{key}: {value}")
            print("----------------------\n")
        print("=========================================\n")

    def print_progress(self, current_index: int, total: int, question: str):
        """Print progress information."""
        progress = ((current_index + 1) / total) * 100
        
        print(f"\n📊 Progress: {current_index + 1}/{total} ({progress:.1f}%)")
        print(f"🔄 Current question: {question[:60]}...")
        print(f"🤖 Using: {self.llm.get_model_info()['provider']} ({self.model_id})")

    def run_chain(self):
        """Run the chain of verification with checkpointing support."""
        # Load existing results from checkpoint if resuming
        all_results = self.checkpoint_data.get("completed_results", [])
        
        try:
            for i in range(self.start_question_index, len(self.questions)):
                question = self.questions[i]
                
                self.print_progress(i, len(self.questions), question)
                
                # Get baseline response
                print("🤖 Generating baseline response...")
                baseline_response = self.get_baseline_response(question)
                
                # Run joint verification (only supported setting)
                print("🔍 Running joint verification...")
                (
                    plan_and_execution_tokens,
                    final_verified_tokens,
                ) = self.run_joint_chain(question, baseline_response)
                result = {
                    "Question": question,
                    "Baseline Answer": baseline_response,
                    "Plan and Execution": plan_and_execution_tokens,
                    "Final Refined Answer": final_verified_tokens,
                }
                
                all_results.append(result)
                self.print_result(result)
                
                # Save checkpoint after each question
                self.save_checkpoint(i, all_results)
            
            # Save final results
            result_file_path = f"result/{self.model_id}_{self.task}_{self.setting}_results.json"
            os.makedirs('result', exist_ok=True)
            with open(result_file_path, "w", encoding="utf-8") as json_file:
                json.dump(all_results, json_file, indent=2, ensure_ascii=False)
            
            print(f"\n🎉 Experiment completed successfully!")
            print(f"📁 Results saved to: {result_file_path}")
            
        except KeyboardInterrupt:
            print(f"\n\n⏹️ Experiment interrupted by user")
            print(f"💾 Progress saved to checkpoint: {self.checkpoint_file}")
            print(f"🔄 Resume with the same command later")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print(f"💾 Progress saved to checkpoint: {self.checkpoint_file}")
            raise e