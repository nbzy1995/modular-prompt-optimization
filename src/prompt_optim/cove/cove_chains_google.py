import os
import json
import time
import sys
from typing import Dict, List, Any
import google.generativeai as genai
from .cove_chains import ChainOfVerification
from ...utils import get_absolute_path


class ChainOfVerificationGoogle(ChainOfVerification):
    def __init__(
        self, model_id, temperature, task, setting, questions, google_access_token
    ):
        super().__init__(model_id, task, setting, questions)
        self.google_access_token = google_access_token
        self.temperature = temperature
        
        # Configure Google AI
        genai.configure(api_key=google_access_token)
        self.model = genai.GenerativeModel(self.model_config.id)
        
        # Rate limiting: 15 requests per minute = 4 seconds between requests
        self.min_request_interval = 4.0
        self.last_request_time = 0
        
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

    def enforce_rate_limit(self):
        """Ensure we don't exceed 15 requests per minute."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            print(f"⏱️ Rate limiting: waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def call_llm(self, prompt: str, max_tokens: int) -> str:
        """Call Google Gemini API with rate limiting and error handling."""
        self.enforce_rate_limit()
        
        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=max_tokens,
            )
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            
            # Extract text from response
            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text.strip()
            else:
                return "No response generated"
                
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                print(f"\n❌ API quota exceeded: {e}")
                print(f"💾 Progress saved to checkpoint: {self.checkpoint_file}")
                print(f"⏰ Please wait and retry with the same command later")
                print(f"   The experiment will automatically resume from where it left off")
                sys.exit(1)
            else:
                print(f"⚠️ API Error: {e}")
                raise e

    def process_prompt(self, prompt: str, command: str) -> str:
        """Process prompt for Google Gemini (no special formatting needed)."""
        return prompt

    def print_progress(self, current_index: int, total: int, question: str):
        """Print progress information."""
        progress = ((current_index + 1) / total) * 100
        remaining = total - (current_index + 1)
        # Estimate time remaining (4 seconds per question for baseline + verification steps)
        # Rough estimate: 3-4 API calls per question depending on setting
        api_calls_per_question = {
            "joint": 2,      # baseline + joint verification
            "two_step": 3,   # baseline + plan + execute + verify
            "factored": 4    # baseline + plan + multiple executes + verify (approximate)
        }.get(self.setting, 3)
        
        estimated_minutes = (remaining * api_calls_per_question * 4) / 60
        
        print(f"\n📊 Progress: {current_index + 1}/{total} ({progress:.1f}%)")
        print(f"🔄 Current question: {question[:60]}...")
        print(f"⏱️ Estimated time remaining: {estimated_minutes:.1f} minutes")

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
                
                # Run verification based on setting
                if self.setting == "two_step":
                    print("🔍 Running two-step verification...")
                    (
                        plan_verification_tokens,
                        execute_verification_tokens,
                        final_verified_tokens,
                    ) = self.run_two_step_chain(question, baseline_response)
                    result = {
                        "Question": question,
                        "Baseline Answer": baseline_response,
                        "Verification Questions": plan_verification_tokens,
                        "Execute Plan": execute_verification_tokens,
                        "Final Refined Answer": final_verified_tokens,
                    }
                elif self.setting == "joint":
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
                elif self.setting == "factored":
                    print("🔍 Running factored verification...")
                    (
                        plan_verification_tokens,
                        execute_verification_tokens,
                        final_verified_tokens,
                    ) = self.run_factored_chain(question, baseline_response)
                    result = {
                        "Question": question,
                        "Baseline Answer": baseline_response,
                        "Verification Questions": plan_verification_tokens,
                        "Execute Plan": execute_verification_tokens,
                        "Final Refined Answer": final_verified_tokens,
                    }
                
                all_results.append(result)
                self.print_result(result)
                
                # Save checkpoint after each question
                self.save_checkpoint(i, all_results)
            
            # Save final results
            result_file_path = (
                f"result/{self.model_id}_{self.task}_{self.setting}_results.json"
            )
            os.makedirs('result', exist_ok=True)
            with open(result_file_path, "w", encoding="utf-8") as json_file:
                json.dump(all_results, json_file, indent=2, ensure_ascii=False)
            
            print(f"\n🎉 Experiment completed successfully!")
            print(f"📁 Results saved to: {result_file_path}")
            
            # Clean up checkpoint
            # self.cleanup_checkpoint()
            
        except KeyboardInterrupt:
            print(f"\n\n⏹️ Experiment interrupted by user")
            print(f"💾 Progress saved to checkpoint: {self.checkpoint_file}")
            print(f"🔄 Resume with the same command: python3 main.py --model={self.model_id} --task={self.task} --setting={self.setting}")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print(f"💾 Progress saved to checkpoint: {self.checkpoint_file}")
            raise e