import unittest
from unittest.mock import patch

from services.sms_service import recognize_sms


class SmsServiceTestCase(unittest.TestCase):
    def test_empty_text_returns_error(self):
        result = recognize_sms("")
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.code, "EMPTY_TEXT")

    @patch("services.sms_service.predict_text")
    def test_normal_sms_returns_low_risk_structure(self, mock_predict_text):
        mock_predict_text.return_value = {
            "prediction": "无风险",
            "probability": 0.12,
            "features": {
                "风险等级": "无风险",
                "关键词": ["无"],
                "链接风险": 0,
                "紧迫性指数": 0,
            },
        }
        result = recognize_sms("这是一条普通提醒短信，不涉及转账和链接。")
        self.assertTrue(result.success)
        self.assertEqual(result.risk_level, "无风险")
        self.assertIsNone(result.fraud_type)
        self.assertIsInstance(result.suggestions, list)

    @patch("services.sms_service.predict_text")
    def test_suspected_fraud_sms_returns_structured_result(self, mock_predict_text):
        mock_predict_text.return_value = {
            "prediction": "冒充电商物流客服类",
            "probability": 0.91,
            "features": {
                "风险等级": "高风险",
                "关键词": ["退款", "验证码"],
                "链接风险": 80,
                "紧迫性指数": 60,
            },
        }
        result = recognize_sms("您好，您的快递异常，请点击链接填写验证码办理退款。")
        self.assertTrue(result.success)
        self.assertEqual(result.risk_level, "高风险")
        self.assertEqual(result.fraud_type, "冒充电商物流客服类")
        self.assertGreater(len(result.evidence), 0)


if __name__ == "__main__":
    unittest.main()
