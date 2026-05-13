import tempfile
import unittest
from unittest.mock import patch

from agent.agent_service import run_agent
from agent.safety import (
    SAFETY_INTENT_EMERGENCY,
    SAFETY_INTENT_NORMAL,
    SAFETY_INTENT_REMEDIAL,
    SAFETY_INTENT_SECONDARY,
    classify_safety_intent,
)
from agent.session_store import AgentSessionStore


class SafetyClassifierTestCase(unittest.TestCase):
    def test_classify_emergency_loss_with_amount(self):
        result = classify_safety_intent("我已经被骗了100万了")
        self.assertEqual(result.safety_intent, SAFETY_INTENT_EMERGENCY)
        self.assertEqual(result.severity, "critical")
        self.assertIn("被骗", result.matched_keywords)

    def test_classify_remedial_action(self):
        result = classify_safety_intent("我已经填写了银行卡号和验证码")
        self.assertEqual(result.safety_intent, SAFETY_INTENT_REMEDIAL)
        self.assertEqual(result.severity, "high")
        self.assertTrue(any("验证码" in item for item in result.matched_keywords))

    def test_classify_secondary_fraud(self):
        result = classify_safety_intent("网警说帮我追回资金，让我交认证费")
        self.assertEqual(result.safety_intent, SAFETY_INTENT_SECONDARY)
        self.assertEqual(result.severity, "high")
        self.assertTrue(any("认证费" in item for item in result.matched_keywords))

    def test_classify_normal_when_empty_or_irrelevant(self):
        self.assertEqual(classify_safety_intent("").safety_intent, SAFETY_INTENT_NORMAL)
        self.assertEqual(classify_safety_intent("你好").safety_intent, SAFETY_INTENT_NORMAL)


class AgentSafetyIntegrationTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = AgentSessionStore(self.temp_dir.name)
        self.store_patcher = patch("agent.agent_service.get_agent_session_store", return_value=self.store)
        self.store_patcher.start()

    def tearDown(self):
        self.store_patcher.stop()
        self.temp_dir.cleanup()

    @patch("agent.agent_service.qa_chat_tool")
    def test_emergency_loss_response_uses_local_template(self, mock_qa):
        response = run_agent({"user_input": "我已经被骗了100万了"})

        self.assertTrue(response.success)
        self.assertEqual(response.safety_intent, SAFETY_INTENT_EMERGENCY)
        self.assertIn(response.risk_level, {"high", "critical"})
        self.assertEqual(response.handler_name, "emergency_loss_handler")
        self.assertIn("报警", response.answer or "")
        self.assertTrue(any("96110" in item for item in response.suggestions))
        self.assertTrue(any(trace.tool_name == "safety_classifier" for trace in response.tool_trace))
        self.assertFalse(mock_qa.called)

        task = self.store.list_task_records(response.session_id)[-1]
        self.assertEqual(task.metadata["safety_intent"], SAFETY_INTENT_EMERGENCY)
        self.assertEqual(task.metadata["handler_name"], "emergency_loss_handler")
        self.assertEqual(task.metadata["error_type"], "SAFETY_EMERGENCY_LOSS")

    @patch("agent.agent_service.qa_chat_tool")
    def test_remedial_action_response_uses_local_template(self, mock_qa):
        response = run_agent({"user_input": "我已经填写了银行卡号和验证码"})

        self.assertTrue(response.success)
        self.assertEqual(response.safety_intent, SAFETY_INTENT_REMEDIAL)
        self.assertEqual(response.risk_level, "high")
        self.assertIn("补救措施", response.conclusion)
        self.assertTrue(any("冻结" in item or "挂失" in item for item in response.suggestions))
        self.assertTrue(any("修改" in item for item in response.suggestions))
        self.assertFalse(mock_qa.called)

    def test_secondary_fraud_response_contains_no_more_transfer_warning(self):
        response = run_agent({"user_input": "有人说能帮我追回被骗的钱，但要先交手续费"})

        self.assertTrue(response.success)
        self.assertEqual(response.safety_intent, SAFETY_INTENT_SECONDARY)
        self.assertEqual(response.risk_level, "high")
        self.assertTrue(any("不要再转账" in item for item in response.suggestions))
        self.assertIn("二次诈骗", response.conclusion)

    def test_contextual_remedial_action_keeps_follow_up_trace(self):
        with patch("agent.agent_service.sms_recognize_tool") as mock_sms, patch(
            "agent.agent_service.case_search_tool"
        ) as mock_case, patch("agent.agent_service.qa_chat_tool") as mock_qa:
            mock_sms.return_value = _tool_result(
                "sms_recognize",
                True,
                data={
                    "risk_level": "high",
                    "fraud_type": "冒充客服诈骗",
                    "confidence": 0.9,
                    "evidence": ["包含异常链接"],
                    "suggestions": ["不要点击链接"],
                },
                summary="完成短信风险识别",
            )
            mock_case.return_value = _tool_result(
                "case_search",
                True,
                data={"cases": [], "total_count": 0},
                summary="检索到 0 条相关案例",
            )
            mock_qa.return_value = _tool_result(
                "qa_chat",
                True,
                data={"answer": "这是高风险诈骗，请勿操作。"},
                summary="问答服务生成了补充建议",
            )

            first = run_agent({"user_input": "帮我看看这条短信是不是诈骗：您的账户异常，请点击链接验证并填写验证码"})
            second = run_agent({"session_id": first.session_id, "user_input": "我已经把验证码告诉他了"})

        self.assertEqual(second.safety_intent, SAFETY_INTENT_REMEDIAL)
        self.assertTrue(any(trace.tool_name == "follow_up_handler" for trace in second.tool_trace))
        self.assertTrue(any(trace.tool_name == "safety_classifier" for trace in second.tool_trace))


def _tool_result(tool_name: str, success: bool, data=None, summary: str = ""):
    from agent.models import AgentToolResult

    return AgentToolResult(tool_name=tool_name, success=success, data=data, summary=summary)


if __name__ == "__main__":
    unittest.main()
