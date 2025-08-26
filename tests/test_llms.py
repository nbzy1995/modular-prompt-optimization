import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llms import LLMProviderFactory, GoogleLLM, ScaledownLLM


class TestLLMProviderFactory(unittest.TestCase):
    """Test the LLM provider factory."""
    
    def test_create_google_provider(self):
        """Test creating Google provider."""
        config = {"GOOGLE_API_KEY": "test-key"}
        
        with patch('llms.genai.configure'), \
             patch('llms.genai.GenerativeModel'):
            
            llm = LLMProviderFactory.create_provider(
                "gemini2.5_flash_lite",
                temperature=0.5,
                configuration=config
            )
            
            self.assertIsInstance(llm, GoogleLLM)
            self.assertEqual(llm.model_id, "gemini2.5_flash_lite")
            self.assertEqual(llm.temperature, 0.5)
    
    def test_create_scaledown_provider(self):
        """Test creating Scaledown provider."""
        config = {"SCALEDOWN_API_KEY": "test-key"}
        
        llm = LLMProviderFactory.create_provider(
            "scaledown-gpt-4o",
            temperature=0.3,
            configuration=config
        )
        
        self.assertIsInstance(llm, ScaledownLLM)
        self.assertEqual(llm.model_id, "scaledown-gpt-4o")
        self.assertEqual(llm.temperature, 0.3)
    
    def test_unsupported_model(self):
        """Test error for unsupported model."""
        with self.assertRaises(ValueError) as context:
            LLMProviderFactory.create_provider("unknown-model")
        
        self.assertIn("Unsupported model", str(context.exception))


class TestGoogleLLM(unittest.TestCase):
    """Test Google LLM provider."""
    
    @patch('llms.genai.configure')
    @patch('llms.genai.GenerativeModel')
    def test_configure_success(self, mock_model, mock_configure):
        """Test successful configuration."""
        config = {"GOOGLE_API_KEY": "test-key"}
        
        llm = GoogleLLM("gemini-1.5-flash", 0.0, config)
        
        mock_configure.assert_called_once_with(api_key="test-key")
        mock_model.assert_called_once_with("gemini-1.5-flash")
    
    def test_configure_missing_key(self):
        """Test configuration fails without API key."""
        config = {}
        
        with self.assertRaises(ValueError) as context:
            GoogleLLM("gemini-1.5-flash", 0.0, config)
        
        self.assertIn("GOOGLE_API_KEY not found", str(context.exception))
    
    @patch('llms.genai.configure')
    @patch('llms.genai.GenerativeModel')
    @patch('llms.time.sleep')
    def test_call_llm_success(self, mock_sleep, mock_model_class, mock_configure):
        """Test successful LLM call."""
        config = {"GOOGLE_API_KEY": "test-key"}
        
        # Mock response
        mock_part = Mock()
        mock_part.text = "Paris is the capital of France."
        mock_candidate = Mock()
        mock_candidate.content.parts = [mock_part]
        mock_response = Mock()
        mock_response.candidates = [mock_candidate]
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        llm = GoogleLLM("gemini-1.5-flash", 0.0, config)
        result = llm.call_llm("What is the capital of France?", 100)
        
        self.assertEqual(result, "Paris is the capital of France.")
        mock_model.generate_content.assert_called_once()
    
    @patch('llms.genai.configure')
    @patch('llms.genai.GenerativeModel')
    def test_model_mapping(self, mock_model_class, mock_configure):
        """Test model name mapping."""
        config = {"GOOGLE_API_KEY": "test-key"}
        
        GoogleLLM("gemini-2.5-flash-lite", 0.0, config)
        
        # Should use model name as-is now
        mock_model_class.assert_called_once_with("gemini-2.5-flash-lite")


class TestScaledownLLM(unittest.TestCase):
    """Test Scaledown LLM provider."""
    
    def test_configure_success(self):
        """Test successful configuration."""
        config = {"SCALEDOWN_API_KEY": "test-key"}
        
        llm = ScaledownLLM("scaledown-gpt-4o", 0.0, config)
        
        self.assertEqual(llm.api_key, "test-key")
        self.assertEqual(llm.actual_model, "gpt-4o")  # Should map scaledown-gpt-4o -> gpt-4o
    
    def test_configure_missing_key(self):
        """Test configuration fails without API key."""
        config = {}
        
        with self.assertRaises(ValueError) as context:
            ScaledownLLM("scaledown-gpt-4o", 0.0, config)
        
        self.assertIn("SCALEDOWN_API_KEY not found", str(context.exception))
    
    @patch('llms.requests.post')
    @patch('llms.time.sleep')
    def test_call_llm_success(self, mock_sleep, mock_post):
        """Test successful LLM call."""
        config = {"SCALEDOWN_API_KEY": "test-key"}
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"full_response": "The answer is 8."}
        mock_post.return_value = mock_response
        
        llm = ScaledownLLM("scaledown-gpt-4o", 0.0, config)
        result = llm.call_llm("What is 5 + 3?", 50)
        
        self.assertEqual(result, "The answer is 8.")
        mock_post.assert_called_once()
        
        # Verify request payload
        call_args = mock_post.call_args
        payload = call_args[1]['data']
        import json
        payload_dict = json.loads(payload)
        self.assertEqual(payload_dict['model'], 'gpt-4o')
        self.assertEqual(payload_dict['prompt'], 'What is 5 + 3?')
    
    @patch('llms.requests.post')
    def test_call_llm_http_error(self, mock_post):
        """Test handling HTTP error."""
        config = {"SCALEDOWN_API_KEY": "test-key"}
        
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        mock_post.return_value = mock_response
        
        llm = ScaledownLLM("scaledown-gpt-4o", 0.0, config)
        
        with self.assertRaises(RuntimeError) as context:
            llm.call_llm("test", 50)
        
        self.assertIn("Scaledown API request failed", str(context.exception))


if __name__ == '__main__':
    unittest.main()