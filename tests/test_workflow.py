import unittest
from unittest.mock import Mock, patch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llms import LLMProviderFactory


class TestSimpleWorkflow(unittest.TestCase):
    """Simple workflow tests that focus on key functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.config = {
            "GOOGLE_API_KEY": "test-google-key",
            "SCALEDOWN_API_KEY": "test-scaledown-key"
        }
    
    @patch('llms.genai.configure')
    @patch('llms.genai.GenerativeModel')
    def test_google_llm_creation_and_call(self, mock_model_class, mock_configure):
        """Test Google LLM creation and basic call."""
        # Mock response
        mock_part = Mock()
        mock_part.text = "Test response from Gemini"
        mock_candidate = Mock()
        mock_candidate.content.parts = [mock_part]
        mock_response = Mock()
        mock_response.candidates = [mock_candidate]
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Create LLM and test
        llm = LLMProviderFactory.create_provider(
            "gemini2.5_flash_lite",
            temperature=0.0,
            configuration=self.config
        )
        
        response = llm.call_llm("Test prompt", 100)
        
        self.assertEqual(response, "Test response from Gemini")
        self.assertEqual(llm.model_id, "gemini2.5_flash_lite")
        mock_configure.assert_called_once_with(api_key="test-google-key")
    
    @patch('llms.requests.post')
    def test_scaledown_llm_creation_and_call(self, mock_post):
        """Test Scaledown LLM creation and basic call."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"full_response": "Test response from Scaledown"}
        mock_post.return_value = mock_response
        
        # Create LLM and test
        llm = LLMProviderFactory.create_provider(
            "scaledown-gpt-4o",
            temperature=0.0,
            configuration=self.config
        )
        
        response = llm.call_llm("Test prompt", 100)
        
        self.assertEqual(response, "Test response from Scaledown")
        self.assertEqual(llm.model_id, "scaledown-gpt-4o")
        self.assertEqual(llm.actual_model, "gpt-4o")  # Model name mapping
    
    def test_model_info_reporting(self):
        """Test that model info is reported correctly."""
        with patch('llms.genai.configure'), \
             patch('llms.genai.GenerativeModel'):
            
            llm = LLMProviderFactory.create_provider(
                "gemini2.5_flash_lite",
                temperature=0.5,
                configuration=self.config
            )
            
            info = llm.get_model_info()
            
            self.assertEqual(info['model_id'], 'gemini2.5_flash_lite')
            self.assertEqual(info['temperature'], 0.5)
            self.assertEqual(info['provider'], 'GoogleLLM')
    
    def test_both_providers_supported(self):
        """Test that both provider types work."""
        with patch('llms.genai.configure'), \
             patch('llms.genai.GenerativeModel'):
            
            google_llm = LLMProviderFactory.create_provider(
                "gemini2.5_flash_lite",
                configuration=self.config
            )
            
            scaledown_llm = LLMProviderFactory.create_provider(
                "scaledown-gpt-4o", 
                configuration=self.config
            )
            
            self.assertEqual(google_llm.__class__.__name__, 'GoogleLLM')
            self.assertEqual(scaledown_llm.__class__.__name__, 'ScaledownLLM')
    
    def test_error_handling(self):
        """Test error handling."""
        # Missing API key
        with self.assertRaises(ValueError):
            LLMProviderFactory.create_provider(
                "gemini2.5_flash_lite",
                configuration={}
            )
        
        # Unsupported model
        with self.assertRaises(ValueError):
            LLMProviderFactory.create_provider(
                "unknown-model",
                configuration=self.config
            )


class TestDataProcessing(unittest.TestCase):
    """Test data processing utilities."""
    
    def test_question_extraction(self):
        """Test question extraction functions."""
        from data.data_processor import get_questions_from_dict, get_questions_from_list
        
        # Test dict format
        dict_data = {"Question 1?": ["answer1"], "Question 2?": ["answer2"]}
        questions = get_questions_from_dict(dict_data)
        self.assertEqual(questions, ["Question 1?", "Question 2?"])
        
        # Test list format
        list_data = [
            {"question": "Question 1?", "answer": "answer1"},
            {"question": "Question 2?", "answer": "answer2"}
        ]
        questions = get_questions_from_list(list_data)
        self.assertEqual(questions, ["Question 1?", "Question 2?"])


if __name__ == '__main__':
    unittest.main()