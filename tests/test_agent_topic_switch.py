import tempfile
import unittest
from unittest.mock import patch

from agent.agent_service import run_agent
from agent.models import AgentToolResult
from agent.safety import SAFETY_INTENT_EMERGENCY, SAFETY_INTENT_SECONDARY
from agent.session_store import AgentSessionStore
from schemas.common_schema import ServiceError


def _tool_result(tool_name: str, success: bool, data=None, summary: str = "", error: ServiceError | None = None):
    return AgentToolResult(
        tool_name=tool_name,
        success=success,
        data=data,
        summary=summary,
        error=error,
    )


class AgentTopicSwitchTestCase(unittest.TestCase):
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
    @patch("agent.agent_service.risk_report_tool")
    def test_user_profile_after_secondary_fraud_is_topic_switch(self, mock_risk, mock_case, mock_qa):
        mock_risk.return_value = _tool_result(
            "risk_report",
            True,
            data={
                "risk_level": "high",
                "vulnerable_fraud_types": [
                    "冒充客服诈骗",
                    "虚假投资理财诈骗",
                    "保健品/养生直播诈骗",
                    "中奖红包福利诈骗",
                    "短视频平台引流诈骗",
                ],
                "reasons": ["60 多岁且高频使用微信和抖音，容易接触社交和直播引流骗局。"],
                "suggestions": [
                    "提醒父母在微信和抖音上遇到陌生客服、中奖福利和投资群时先暂停操作。",
                    "和父母约定，凡是直播促销、养生推荐、投资拉群都先和家人确认。",
                    "看到要转账、点链接、加好友的内容时，先通过官方渠道核实。",
                ],
                "report": "",
            },
            summary="完成用户风险评估",
        )
        mock_case.return_value = _tool_result(
            "case_search",
            True,
            data={"cases": [{"title": "老年人直播保健品诈骗", "summary": "通过养生直播诱导下单", "source": "neo4j"}], "total_count": 1},
            summary="检索到 1 条相关案例",
        )
        mock_qa.return_value = _tool_result(
            "qa_chat",
            True,
            data={
                "answer": "父母如果常用微信和抖音，要重点防范冒充客服、虚假投资理财、保健品/养生直播、中奖红包福利和短视频引流骗局。"
            },
            summary="完成建议生成",
        )

        first = run_agent({"user_input": "有人说能帮我追回被骗的钱，但要先交手续费"})
        second = run_agent(
            {
                "session_id": first.session_id,
                "user_input": "那我爸妈60多岁，经常用微信和抖音，容易遇到什么骗局？",
                "options": {"return_trace": True, "case_limit": 5, "use_memory": True, "history_limit": 6},
            }
        )

        self.assertEqual(second.task_type, "user_risk_profile")
        self.assertNotIn("手续费", second.answer or "")
        self.assertNotIn("追回", second.answer or "")
        self.assertTrue(any(keyword in " ".join(second.suggestions) for keyword in ("微信", "抖音", "父母", "老人")))
        task = self.store.list_task_records(second.session_id)[-1]
        self.assertTrue(task.metadata["topic_switch_detected"])
        self.assertEqual(task.metadata["suggested_task_type"], "user_risk_profile")
        self.assertEqual(task.metadata["previous_task_type"], first.task_type)
        self.assertFalse(task.metadata["follow_up_detected"])

    @patch("agent.agent_service.qa_chat_tool")
    @patch("agent.agent_service.graph_query_tool")
    @patch("agent.agent_service.case_search_tool")
    @patch("agent.agent_service.sms_recognize_tool")
    def test_case_summary_after_high_risk_task_is_topic_switch(self, mock_sms, mock_case, mock_graph, mock_qa):
        mock_sms.return_value = _tool_result(
            "sms_recognize",
            True,
            data={
                "risk_level": "high",
                "fraud_type": "冒充客服诈骗",
                "confidence": 0.92,
                "evidence": ["要求点击退款链接"],
                "suggestions": ["不要点击链接"],
            },
            summary="完成短信风险识别",
        )
        mock_case.side_effect = [
            _tool_result("case_search", True, data={"cases": [], "total_count": 0}, summary="无结果"),
            _tool_result(
                "case_search",
                True,
                data={"cases": [{"title": "杀猪盘案例", "summary": "先培养感情再诱导投资", "source": "neo4j"}], "total_count": 1},
                summary="检索到 1 条案例",
            ),
        ]
        mock_graph.return_value = _tool_result(
            "graph_query",
            True,
            data={"explanation": "常见链路是社交引流、建立信任、再导向虚假投资平台。"},
            summary="完成图谱查询",
        )
        mock_qa.side_effect = [
            _tool_result("qa_chat", True, data={"answer": "这是高风险退款诈骗。"}, summary="首轮建议"),
            _tool_result("qa_chat", True, data={"answer": "杀猪盘通常先养熟关系，再诱导受害者投资。"}, summary="案例总结"),
        ]

        first = run_agent({"user_input": "帮我看看这条短信是不是诈骗：客服说快递丢了，让我点链接退款。"})
        second = run_agent(
            {
                "session_id": first.session_id,
                "user_input": "杀猪盘一般有什么特征？",
                "options": {"return_trace": True, "case_limit": 5, "use_memory": True, "history_limit": 6},
            }
        )

        self.assertEqual(second.task_type, "case_summary")
        task = self.store.list_task_records(second.session_id)[-1]
        self.assertTrue(task.metadata["topic_switch_detected"])
        self.assertEqual(task.metadata["suggested_task_type"], "case_summary")

    @patch("agent.agent_service.qa_chat_tool")
    @patch("agent.agent_service.case_search_tool")
    @patch("agent.agent_service.sms_recognize_tool")
    @patch("agent.agent_service.risk_report_tool")
    def test_new_sms_after_profile_is_topic_switch(self, mock_risk, mock_sms, mock_case, mock_qa):
        mock_risk.return_value = _tool_result(
            "risk_report",
            True,
            data={
                "risk_level": "medium",
                "vulnerable_fraud_types": ["冒充客服诈骗"],
                "reasons": ["常用购物和社交平台。"],
                "suggestions": ["遇到客服退款先通过官方渠道核实。"],
                "report": "",
            },
            summary="完成用户风险评估",
        )
        mock_sms.return_value = _tool_result(
            "sms_recognize",
            True,
            data={
                "risk_level": "high",
                "fraud_type": "冒充客服诈骗",
                "confidence": 0.95,
                "evidence": ["包含账户异常和陌生链接"],
                "suggestions": ["不要点击陌生链接"],
            },
            summary="完成短信识别",
        )
        mock_case.side_effect = [
            _tool_result("case_search", True, data={"cases": [], "total_count": 0}, summary="画像阶段无结果"),
            _tool_result(
                "case_search",
                True,
                data={"cases": [{"title": "账户异常短信诈骗", "summary": "诱导点击链接验证", "source": "neo4j"}], "total_count": 1},
                summary="检索到 1 条案例",
            ),
        ]
        mock_qa.side_effect = [
            _tool_result("qa_chat", True, data={"answer": "这类人群要防范退款和客服类骗局。"}, summary="画像建议"),
            _tool_result("qa_chat", True, data={"answer": "这条短信存在明显诈骗风险，不要点击链接验证。"}, summary="短信建议"),
        ]

        first = run_agent({"user_input": "我是大学生，经常网购，容易遇到什么骗局？"})
        second = run_agent(
            {
                "session_id": first.session_id,
                "user_input": "帮我看看这条新短信是不是诈骗：您的账户异常，请点击链接验证。",
                "options": {"return_trace": True, "case_limit": 5, "use_memory": True, "history_limit": 6},
            }
        )

        self.assertEqual(second.task_type, "text_risk_analysis")
        task = self.store.list_task_records(second.session_id)[-1]
        self.assertTrue(task.metadata["topic_switch_detected"])
        self.assertEqual(task.metadata["suggested_task_type"], "text_risk_analysis")

    @patch("agent.agent_service.qa_chat_tool")
    @patch("agent.agent_service.case_search_tool")
    @patch("agent.agent_service.sms_recognize_tool")
    def test_short_next_step_still_follow_up(self, mock_sms, mock_case, mock_qa):
        mock_sms.return_value = _tool_result(
            "sms_recognize",
            True,
            data={
                "risk_level": "high",
                "fraud_type": "冒充客服诈骗",
                "confidence": 0.93,
                "evidence": ["要求点击链接退款"],
                "suggestions": ["不要点击链接"],
            },
            summary="完成短信风险识别",
        )
        mock_case.return_value = _tool_result("case_search", True, data={"cases": [], "total_count": 0}, summary="无结果")
        mock_qa.side_effect = [
            _tool_result("qa_chat", True, data={"answer": "这是高风险诈骗短信。"}, summary="首轮建议"),
            _tool_result("qa_chat", True, data={"answer": "现在先停止操作，再通过官方客服核实，并尽快修改相关账户密码。"}, summary="追问建议"),
        ]

        first = run_agent({"user_input": "帮我看看这条短信是不是诈骗：客服让我点链接退款。"})
        second = run_agent(
            {
                "session_id": first.session_id,
                "user_input": "那我现在应该怎么办？",
                "options": {"return_trace": True, "case_limit": 5, "use_memory": True, "history_limit": 6},
            }
        )

        self.assertTrue(second.context_used)
        self.assertEqual(second.task_type, "text_risk_analysis")
        task = self.store.list_task_records(second.session_id)[-1]
        self.assertFalse(task.metadata["topic_switch_detected"])
        self.assertTrue(task.metadata["follow_up_detected"])
        self.assertIn("停止操作", second.answer or "")

    @patch("agent.agent_service.qa_chat_tool")
    @patch("agent.agent_service.case_search_tool")
    @patch("agent.agent_service.risk_report_tool")
    def test_emergency_loss_still_safety_first(self, mock_risk, mock_case, mock_qa):
        mock_risk.return_value = _tool_result(
            "risk_report",
            True,
            data={
                "risk_level": "medium",
                "vulnerable_fraud_types": ["刷单返利诈骗"],
                "reasons": ["学生容易被兼职诱惑。"],
                "suggestions": ["凡是先垫付的兼职都不要做。"],
                "report": "",
            },
            summary="完成用户风险评估",
        )
        mock_case.return_value = _tool_result("case_search", True, data={"cases": [], "total_count": 0}, summary="无结果")
        mock_qa.return_value = _tool_result("qa_chat", True, data={"answer": "要防范兼职垫付骗局。"}, summary="画像建议")

        first = run_agent({"user_input": "我是大学生，经常找兼职，容易遇到什么骗局？"})
        second = run_agent(
            {
                "session_id": first.session_id,
                "user_input": "我已经被骗了100万了",
                "options": {"return_trace": True, "case_limit": 5, "use_memory": True, "history_limit": 6},
            }
        )

        self.assertEqual(second.safety_intent, SAFETY_INTENT_EMERGENCY)
        task = self.store.list_task_records(second.session_id)[-1]
        self.assertEqual(task.metadata["safety_intent"], SAFETY_INTENT_EMERGENCY)

    def test_secondary_fraud_still_safety_first(self):
        response = run_agent({"user_input": "有人说能帮我追回被骗的钱，但要先交手续费"})

        self.assertEqual(response.safety_intent, SAFETY_INTENT_SECONDARY)
        self.assertTrue(any("不要再转账" in item for item in response.suggestions))


if __name__ == "__main__":
    unittest.main()
