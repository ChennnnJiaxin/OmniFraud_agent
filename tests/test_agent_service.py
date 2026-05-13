import unittest
from unittest.mock import patch

from agent.agent_service import run_agent
from agent.models import AgentRunRequest, AgentToolResult
from schemas.common_schema import ServiceError


def _tool_result(tool_name: str, success: bool, data=None, summary: str = "", error: ServiceError | None = None):
    return AgentToolResult(
        tool_name=tool_name,
        success=success,
        data=data,
        summary=summary,
        error=error,
    )


class AgentServiceTestCase(unittest.TestCase):
    @patch("agent.agent_service.qa_chat_tool")
    @patch("agent.agent_service.case_search_tool")
    @patch("agent.agent_service.sms_recognize_tool")
    def test_text_sms_risk_analysis(self, mock_sms, mock_case, mock_qa):
        mock_sms.return_value = _tool_result(
            "sms_recognize",
            True,
            data={
                "risk_level": "high",
                "fraud_type": "冒充客服诈骗",
                "confidence": 0.91,
                "evidence": ["包含异常链接", "制造紧迫感"],
                "suggestions": ["不要点击陌生链接"],
            },
            summary="完成短信风险识别",
        )
        mock_case.return_value = _tool_result(
            "case_search",
            True,
            data={
                "cases": [
                    {"title": "冒充客服退款诈骗案例", "summary": "冒充客服诱导转账", "source": "neo4j"},
                ],
                "total_count": 1,
            },
            summary="检索到 1 条相似案例",
        )
        mock_qa.return_value = _tool_result(
            "qa_chat",
            True,
            data={"answer": "建议通过官方客服电话核实，不要提供验证码。"},
            summary="问答服务生成了补充建议",
        )

        response = run_agent({"user_input": "帮我看看这段短信是不是诈骗：您的账户异常，请点击链接验证"})

        self.assertTrue(response.success)
        self.assertEqual(response.task_type, "text_risk_analysis")
        self.assertEqual(response.fraud_type, "冒充客服诈骗")
        self.assertEqual(len(response.related_cases), 1)
        self.assertGreaterEqual(len(response.tool_trace), 3)

    @patch("agent.agent_service.qa_chat_tool")
    @patch("agent.agent_service.case_search_tool")
    @patch("agent.agent_service.sms_recognize_tool")
    def test_customer_service_refund_analysis_uses_fraud_type_for_case_search(self, mock_sms, mock_case, mock_qa):
        mock_sms.return_value = _tool_result(
            "sms_recognize",
            True,
            data={
                "risk_level": "high",
                "fraud_type": "冒充客服诈骗",
                "confidence": 0.88,
                "evidence": ["提到快递退款"],
                "suggestions": ["不要添加陌生 QQ"],
            },
            summary="完成短信风险识别",
        )
        mock_case.return_value = _tool_result("case_search", True, data={"cases": [], "total_count": 0}, summary="无结果")
        mock_qa.return_value = _tool_result("qa_chat", True, data={"answer": "请通过官方平台核实。"}, summary="完成建议")

        run_agent({"user_input": "这是不是冒充客服诈骗？他说我快递丢了，要我加 QQ 办理退款。"})

        _, kwargs = mock_case.call_args
        self.assertEqual(kwargs["fraud_type"], "冒充客服诈骗")

    @patch("agent.agent_service.qa_chat_tool")
    @patch("agent.agent_service.case_search_tool")
    @patch("agent.agent_service.sms_recognize_tool")
    def test_brushing_order_analysis_returns_high_risk(self, mock_sms, mock_case, mock_qa):
        mock_sms.return_value = _tool_result(
            "sms_recognize",
            True,
            data={
                "risk_level": "high",
                "fraud_type": "刷单返利诈骗",
                "confidence": 0.95,
                "evidence": ["存在垫付要求"],
                "suggestions": ["不要先行垫付"],
            },
            summary="完成短信风险识别",
        )
        mock_case.return_value = _tool_result("case_search", True, data={"cases": [], "total_count": 0}, summary="无结果")
        mock_qa.return_value = _tool_result("qa_chat", True, data={"answer": "刷单返利通常以小额返利诱导大额投入。"}, summary="完成建议")

        response = run_agent({"user_input": "兼职刷单群里有人让我垫付，这靠谱吗？"})

        self.assertEqual(response.task_type, "text_risk_analysis")
        self.assertEqual(response.risk_level, "high")
        self.assertEqual(response.fraud_type, "刷单返利诈骗")

    @patch("agent.agent_service.qa_chat_tool")
    @patch("agent.agent_service.graph_query_tool")
    @patch("agent.agent_service.case_search_tool")
    def test_pig_butchering_case_summary(self, mock_case, mock_graph, mock_qa):
        mock_case.return_value = _tool_result(
            "case_search",
            True,
            data={
                "cases": [{"title": "杀猪盘典型案例", "summary": "长期情感铺垫后诱导投资", "source": "neo4j"}],
                "total_count": 1,
            },
            summary="检索到 1 条相似案例",
        )
        mock_graph.return_value = _tool_result(
            "graph_query",
            True,
            data={"explanation": "图谱显示常见链路为社交引流-恋爱关系-投资平台。"},
            summary="完成图谱查询",
        )
        mock_qa.return_value = _tool_result(
            "qa_chat",
            True,
            data={"answer": "杀猪盘通常先建立信任，再引导受害者进入虚假投资平台。"},
            summary="完成案例总结",
        )

        response = run_agent({"user_input": "杀猪盘一般有什么特征？"})

        self.assertTrue(response.success)
        self.assertEqual(response.task_type, "case_summary")
        self.assertIsNotNone(response.graph_result)
        self.assertIn("杀猪盘", response.answer)

    @patch("agent.agent_service.qa_chat_tool")
    @patch("agent.agent_service.graph_query_tool")
    @patch("agent.agent_service.case_search_tool")
    def test_similar_case_query_summary(self, mock_case, mock_graph, mock_qa):
        mock_case.return_value = _tool_result(
            "case_search",
            True,
            data={"cases": [{"title": "冒充客服案例", "summary": "退款诱导转账", "source": "neo4j"}], "total_count": 3},
            summary="检索到 1 条相似案例",
        )
        mock_graph.return_value = _tool_result(
            "graph_query",
            False,
            error=ServiceError(code="GRAPH_QUERY_FAILED", message="图谱查询失败"),
            summary="图谱查询失败",
        )
        mock_qa.return_value = _tool_result(
            "qa_chat",
            True,
            data={"answer": "常见套路包括冒充平台客服、编造退款理由和诱导屏幕共享。"},
            summary="完成案例总结",
        )

        response = run_agent({"user_input": "有没有冒充客服退款的类似案例？"})

        self.assertTrue(response.success)
        self.assertEqual(response.task_type, "case_summary")
        self.assertEqual(len(response.related_cases), 1)
        self.assertIsNone(response.graph_result)

    @patch("agent.agent_service.qa_chat_tool")
    @patch("agent.agent_service.case_search_tool")
    @patch("agent.agent_service.risk_report_tool")
    def test_elderly_user_profile_risk_suggestion(self, mock_risk, mock_case, mock_qa):
        mock_risk.return_value = _tool_result(
            "risk_report",
            True,
            data={
                "risk_level": "medium",
                "vulnerable_fraud_types": ["保健品诈骗", "冒充客服诈骗"],
                "reasons": ["年龄较大且高频使用社交平台"],
                "suggestions": ["遇到保健品推销先与家人核实"],
                "report": "",
            },
            summary="完成用户风险评估",
        )
        mock_case.return_value = _tool_result(
            "case_search",
            True,
            data={"cases": [{"title": "保健品诈骗案例", "summary": "诱导老人高价购买", "source": "neo4j"}], "total_count": 1},
            summary="检索到 1 条相似案例",
        )
        mock_qa.return_value = _tool_result(
            "qa_chat",
            True,
            data={"answer": "这类人群容易被情感关怀、健康焦虑和退款承诺打动。"},
            summary="完成建议",
        )

        response = run_agent({"user_input": "我爸妈60多岁，经常用微信和抖音，容易遇到什么骗局？"})

        self.assertTrue(response.success)
        self.assertEqual(response.task_type, "user_risk_profile")
        self.assertIn("保健品诈骗", response.vulnerable_fraud_types)

    @patch("agent.agent_service.qa_chat_tool")
    @patch("agent.agent_service.case_search_tool")
    @patch("agent.agent_service.risk_report_tool")
    def test_student_brushing_order_profile_suggestion(self, mock_risk, mock_case, mock_qa):
        mock_risk.return_value = _tool_result(
            "risk_report",
            True,
            data={
                "risk_level": "high",
                "vulnerable_fraud_types": ["刷单返利诈骗"],
                "reasons": ["学生群体易被兼职高回报吸引"],
                "suggestions": ["任何先垫付的兼职都要警惕"],
                "report": "",
            },
            summary="完成用户风险评估",
        )
        mock_case.return_value = _tool_result(
            "case_search",
            True,
            data={"cases": [{"title": "刷单返利案例", "summary": "先返小额再骗大额", "source": "neo4j"}], "total_count": 1},
            summary="检索到 1 条相似案例",
        )
        mock_qa.return_value = _tool_result(
            "qa_chat",
            True,
            data={"answer": "兼职群要求先打款、先垫付、先做任务，都是高危信号。"},
            summary="完成建议",
        )

        response = run_agent({"user_input": "我是大学生，经常兼职刷单群里有人让我垫付，风险大吗？"})

        self.assertTrue(response.success)
        self.assertEqual(response.task_type, "user_risk_profile")
        self.assertEqual(response.risk_level, "high")

    def test_empty_input_returns_error(self):
        response = run_agent({"user_input": "   "})
        self.assertFalse(response.success)
        self.assertEqual(response.task_type, "unknown")
        self.assertEqual(response.error.code, "EMPTY_INPUT")

    def test_greeting_returns_out_of_scope_intro(self):
        response = run_agent({"user_input": "你好啊"})
        self.assertTrue(response.success)
        self.assertEqual(response.task_type, "out_of_scope")
        self.assertIn("反诈助手", response.answer or "")
        self.assertIn("诈骗", response.answer or "")

    @patch("agent.agent_service.ocr_tool")
    def test_ocr_unavailable_returns_fallback_result(self, mock_ocr):
        mock_ocr.return_value = _tool_result(
            "ocr",
            False,
            error=ServiceError(code="OCR_FAILED", message="OCR 不可用"),
            summary="OCR 识别失败",
        )

        response = run_agent(AgentRunRequest(user_input="帮我分析这张截图有没有诈骗风险", input_type="image", image="fake-image"))

        self.assertFalse(response.success)
        self.assertEqual(response.task_type, "image_risk_analysis")
        self.assertEqual(response.error.code, "OCR_FAILED")
        self.assertIn("手动输入", response.fallback_message)


if __name__ == "__main__":
    unittest.main()
