import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.analyzer import _ACTION_PATTERNS

class TestNLPProcessor(unittest.TestCase):
    
    def test_extract_action_items_regex(self):
        text1 = "We should definitely schedule a meeting for tomorrow."
        text2 = "Please submit the report by Friday."
        text3 = "Action item: review the pull request."
        text4 = "The sky is blue."
        
        self.assertTrue(_ACTION_PATTERNS.search(text1), "Failed to match 'schedule'")
        self.assertTrue(_ACTION_PATTERNS.search(text2), "Failed to match 'submit'")
        self.assertTrue(_ACTION_PATTERNS.search(text3), "Failed to match 'action item'")
        self.assertFalse(_ACTION_PATTERNS.search(text4), "Matched non-action text")

if __name__ == '__main__':
    unittest.main()
