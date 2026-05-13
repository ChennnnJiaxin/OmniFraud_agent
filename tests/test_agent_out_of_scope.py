import tempfile
import unittest
from unittest.mock import patch

from agent.agent_service import run_agent
from agent.models import AgentToolResult
from agent.session_store import AgentSessionStore


def _tool_result(tool_name: str, success: bool, data=None, summary: str = ""):
    return AgentToolResult(tool_name=tool_name, success=success, data=data, summary=summary)


class AgentOutOfScopeTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = AgentSessionStore(self.temp_dir.name)
        self.store_patcher = patch("agent.agent_service.get_agent_session_store", return_value=self.store)
        self.store_patcher.start()

    def tearDown(self):
        self.store_patcher.stop()
        self.temp_dir.cleanup()

    @patch("agent.agent_service.qa_chat_tool")
    @patch("agent.agent_service.case_search_tool")
    @patch("agent.agent_service.sms_recognize_tool")
    def test_math_question_is_out_of_scope(self, mock_sms, mock_case, mock_qa):
        response = run_agent({"user_input": "1+1等于几"})

        self.assertTrue(response.success)
        self.assertEqual(response.task_type, "out_of_scope")
        self.assertNotEqual(response.task_type, "unknown")
        self.assertIn("反诈助手", response.answer or "")
        self.assertTrue("短信" in (response.answer or "") or "诈骗" in (response.answer or ""))
        self.assertFalse(mock_sms.called)
        self.assertFalse(mock_case.called)
        self.assertFalse(mock_qa.called)
        self.assertTrue(any(trace.tool_name == "out_of_scope_detector" for trace in response.tool_trace))

    def test_weather_question_is_out_of_scope(self):
        response = run_agent({"user_input": "今天天气怎么样"})
        self.assertEqual(response.task_type, "out_of_scope")

    def test_code_question_is_out_of_scope(self):
        response = run_agent({"user_input": "Python 怎么写冒泡排序"})
        self.assertEqual(response.task_type, "out_of_scope")

    def test_greeting_is_intro_not_unknown(self):
        response = run_agent({"user_input": "你好"})
        self.assertEqual(response.task_type, "out_of_scope")
        self.assertIn("反诈助手", response.answer or "")

    def test_vague_possible_fraud_still_unknown(self):
        response = run_agent({"user_input": "帮我看看这个"})
        self.assertEqual(response.task_type, "unknown")
        self.assertNotEqual(response.task_type, "out_of_scope")

    @patch("agent.agent_service.qa_chat_tool")
    @patch("agent.agent_service.case_search_tool")
    @patch("agent.agent_service.risk_report_tool")
    def test_parent_profile_not_out_of_scope(self, mock_risk, mock_case, mock_qa):
        mock_risk.return_value = _tool_result(
            "risk_report",
            True,
            data={
                "risk_level": "medium",
                "vulnerable_fraud_types": ["保健品诈骗"],
                "reasons": ["老年人高频使用社交和短视频平台"],
                "suggestions": ["涉及转账前先和家人核实"],
                "report": "",
            },
            summary="完成用户风险评估",
        )
        mock_case.return_value = _tool_result("case_search", True, data={"cases": [], "total_count": 0}, summary="无结果")
        mock_qa.return_value = _tool_result("qa_chat", True, data={"answer": "需要重点防范养生直播和冒充客服骗局。"}, summary="完成建议")

        response = run_agent({"user_input": "我爸妈60多岁，经常用微信和抖音，容易遇到什么骗局？"})

        self.assertEqual(response.task_type, "user_risk_profile")
        self.assertNotEqual(response.task_type, "out_of_scope")

    def test_emergency_loss_not_out_of_scope(self):
        response = run_agent({"user_input": "我已经被骗了100万了"})
        self.assertEqual(response.safety_intent, "emergency_loss")
        self.assertNotEqual(response.task_type, "out_of_scope")

    def test_secondary_fraud_not_out_of_scope(self):
        response = run_agent({"user_input": "有人说能帮我追回被骗的钱，但要先交手续费"})
        self.assertEqual(response.safety_intent, "secondary_fraud")
        self.assertNotEqual(response.task_type, "out_of_scope")

    @patch("agent.agent_service.qa_chat_tool")
    @patch("agent.agent_service.case_search_tool")
    @patch("agent.agent_service.sms_recognize_tool")
    def test_out_of_scope_does_not_follow_previous_fraud_context(self, mock_sms, mock_case, mock_qa):
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
        mock_case.return_value = _tool_result("case_search", True, data={"cases": [], "total_count": 0}, summary="无结果")
        mock_qa.return_value = _tool_result("qa_chat", True, data={"answer": "这是高风险诈骗短信。"}, summary="完成建议")

        first = run_agent({"user_input": "帮我看看这条短信是不是诈骗：您的账户异常，请点击链接验证"})
        second = run_agent({"session_id": first.session_id, "user_input": "今天天气怎么样"})

        self.assertEqual(second.task_type, "out_of_scope")
        self.assertFalse(any(trace.tool_name == "follow_up_handler" for trace in second.tool_trace))
        task = self.store.list_task_records(second.session_id)[-1]
        self.assertFalse(task.metadata["follow_up_detected"])
        self.assertEqual(task.metadata["out_of_scope_reason"], "用户输入是天气、时间或新闻查询，与反诈任务无关")


if __name__ == "__main__":
    unittest.main()
