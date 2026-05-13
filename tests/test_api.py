import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from agent.models import AgentRunResponse, AgentToolTrace
from agent.session_models import AgentSession
from api.main import app
from schemas.case_schema import CaseItem, CaseSearchResponse
from schemas.common_schema import ServiceError
from schemas.qa_schema import ChatResponse
from schemas.risk_schema import RiskReportResponse
from schemas.sms_schema import SmsRecognizeResponse


class ApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, raise_server_exceptions=False)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "success": True,
                "service": "OmniFraud API",
                "status": "ok",
            },
        )

    @patch("api.routes.sms_routes.get_sms_service")
    def test_sms_recognize_success(self, mock_get_service):
        mock_get_service.return_value = Mock(
            return_value=SmsRecognizeResponse(
                success=True,
                risk_level="high",
                fraud_type="冒充客服诈骗",
                confidence=0.86,
                evidence=["包含异常链接"],
                suggestions=["不要点击链接"],
                raw_result={"prediction": "冒充客服诈骗"},
            )
        )
        response = self.client.post("/sms/recognize", json={"text": "您好，您的快递异常，请点击链接办理退款。"})
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(body["success"])
        self.assertEqual(body["risk_level"], "high")
        self.assertEqual(body["fraud_type"], "冒充客服诈骗")

    def test_sms_recognize_empty_text(self):
        response = self.client.post("/sms/recognize", json={"text": "   "})
        body = response.json()
        self.assertEqual(response.status_code, 400)
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], "INVALID_INPUT")

    @patch("api.routes.qa_routes.get_qa_service")
    def test_qa_chat_success(self, mock_get_service):
        mock_get_service.return_value = Mock(
            return_value=ChatResponse(
                success=True,
                answer="杀猪盘通常具有长期情感铺垫、诱导投资、虚假平台交易等特征。",
            )
        )
        response = self.client.post("/qa/chat", json={"question": "杀猪盘一般有什么特征？", "context": None})
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(body["success"])
        self.assertIn("杀猪盘", body["answer"])
        self.assertEqual(body["references"], [])

    def test_qa_chat_empty_question(self):
        response = self.client.post("/qa/chat", json={"question": ""})
        body = response.json()
        self.assertEqual(response.status_code, 400)
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], "INVALID_INPUT")

    @patch("api.routes.case_routes.get_case_service")
    def test_case_search_success(self, mock_get_service):
        service_callable = Mock(
            return_value=CaseSearchResponse(
                success=True,
                cases=[
                    CaseItem(
                        title="冒充快递客服退款诈骗案例",
                        summary="诈骗分子冒充快递客服，以快递丢失退款为由诱导受害者转账。",
                        source="知识库",
                    )
                ],
                total_count=1,
            )
        )
        mock_get_service.return_value = service_callable
        response = self.client.post(
            "/cases/search",
            json={
                "query": "冒充客服退款",
                "fraud_type": "冒充客服诈骗",
                "limit": 5,
                "page": 1,
                "page_size": 5,
            },
        )
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(body["success"])
        self.assertEqual(body["total"], 1)
        self.assertEqual(body["page_size"], 5)
        service_callable.assert_called_once_with(
            query="冒充客服退款",
            fraud_type="冒充客服诈骗",
            limit=5,
            skip=0,
        )

    def test_case_search_empty_query(self):
        response = self.client.post("/cases/search", json={"query": " "})
        body = response.json()
        self.assertEqual(response.status_code, 400)
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], "INVALID_INPUT")

    @patch("api.routes.risk_routes.get_risk_service")
    def test_risk_report_success(self, mock_get_service):
        mock_get_service.return_value = Mock(
            return_value=RiskReportResponse(
                success=True,
                risk_level="medium",
                vulnerable_fraud_types=["冒充客服诈骗", "中奖诈骗"],
                reasons=["高频使用社交和电商平台"],
                suggestions=["不要点击陌生链接"],
            )
        )
        response = self.client.post(
            "/risk-assessment/report",
            json={
                "age": 62,
                "occupation": "退休",
                "platforms": ["微信", "抖音", "拼多多"],
                "recent_experience": "经常收到中奖、退款、投资理财类消息",
                "risk_preference": "容易相信熟人推荐",
            },
        )
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(body["success"])
        self.assertEqual(body["risk_level"], "medium")
        self.assertGreater(len(body["suggestions"]), 0)

    def test_risk_report_insufficient_info(self):
        response = self.client.post("/risk-assessment/report", json={"risk_preference": "容易相信熟人推荐"})
        body = response.json()
        self.assertEqual(response.status_code, 400)
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], "INVALID_INPUT")

    @patch("api.routes.agent_routes.run_agent")
    def test_agent_run_success(self, mock_run_agent):
        mock_run_agent.return_value = AgentRunResponse(
            success=True,
            task_type="text_risk_analysis",
            conclusion="该内容存在较高诈骗风险",
            risk_level="high",
            fraud_type="冒充客服诈骗",
            confidence=0.86,
            evidence=["包含异常链接"],
            suggestions=["不要点击链接"],
            answer="请通过官方渠道核实。",
            tool_trace=[AgentToolTrace(tool_name="sms_recognize", success=True, summary="完成短信风险识别")],
        )
        response = self.client.post(
            "/agent/run",
            json={
                "user_input": "帮我看看这段短信是不是诈骗：您的账户异常，请点击链接验证",
                "input_type": "text",
                "options": {"return_trace": True, "case_limit": 5},
            },
        )
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(body["success"])
        self.assertEqual(body["task_type"], "text_risk_analysis")
        self.assertEqual(body["fraud_type"], "冒充客服诈骗")

    def test_agent_run_empty_user_input(self):
        response = self.client.post(
            "/agent/run",
            json={
                "user_input": "   ",
                "input_type": "text",
                "options": {"return_trace": True, "case_limit": 5},
            },
        )
        body = response.json()
        self.assertEqual(response.status_code, 400)
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], "INVALID_INPUT")

    @patch("api.routes.sms_routes.get_sms_service")
    def test_service_exception_returns_internal_error(self, mock_get_service):
        service_callable = Mock(side_effect=RuntimeError("boom"))
        mock_get_service.return_value = service_callable
        response = self.client.post("/sms/recognize", json={"text": "您好，您的快递异常，请点击链接办理退款。"})
        body = response.json()
        self.assertEqual(response.status_code, 500)
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], "INTERNAL_ERROR")

    @patch("api.routes.qa_routes.get_qa_service")
    def test_service_error_payload_is_normalized(self, mock_get_service):
        mock_get_service.return_value = Mock(
            return_value=ChatResponse(
                success=False,
                error=ServiceError(code="QA_FAILED", message="问答服务暂时不可用", detail={"reason": "mock"}),
            )
        )
        response = self.client.post("/qa/chat", json={"question": "诈骗怎么防？"})
        body = response.json()
        self.assertEqual(response.status_code, 502)
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], "LLM_ERROR")
        self.assertEqual(body["error"]["detail"], {"reason": "mock"})

    @patch("api.routes.session_routes.get_agent_session_store")
    def test_list_sessions_route_returns_recent_sessions(self, mock_get_store):
        mock_store = Mock()
        mock_store.list_sessions.return_value = [
            AgentSession(
                session_id="session-new",
                created_at="2026-05-04T10:00:00Z",
                updated_at="2026-05-04T12:00:00Z",
                title="新的分析",
                metadata={},
            ),
            AgentSession(
                session_id="session-old",
                created_at="2026-05-04T09:00:00Z",
                updated_at="2026-05-04T09:30:00Z",
                title="旧会话",
                metadata={},
            ),
        ]
        mock_get_store.return_value = mock_store

        response = self.client.get("/agent/sessions", params={"limit": 20})
        body = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(body["success"])
        self.assertEqual(len(body["sessions"]), 2)
        self.assertEqual(body["sessions"][0]["session_id"], "session-new")
        self.assertEqual(body["sessions"][1]["session_id"], "session-old")
        mock_store.list_sessions.assert_called_once_with(limit=20)


if __name__ == "__main__":
    unittest.main()
