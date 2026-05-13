import unittest
from unittest.mock import patch

from services.qa_service import chat_with_anti_fraud_bot


class _FakeAgent:
    def invoke(self, _payload, _config):
        return {"output": "请不要点击陌生链接，并通过官方渠道核验信息。"}


class QaServiceTestCase(unittest.TestCase):
    def test_empty_question_returns_error(self):
        result = chat_with_anti_fraud_bot("")
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.code, "EMPTY_QUESTION")

    @patch("services.qa_service._build_agent")
    def test_normal_question_returns_answer(self, mock_build_agent):
        mock_build_agent.return_value = _FakeAgent()
        result = chat_with_anti_fraud_bot("收到陌生退款短信怎么办？", context={"session_id": "test"})
        self.assertTrue(result.success)
        self.assertIn("不要点击陌生链接", result.answer)


if __name__ == "__main__":
    unittest.main()
