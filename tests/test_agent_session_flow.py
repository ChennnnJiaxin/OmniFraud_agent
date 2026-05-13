import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from agent.agent_service import run_agent
from agent.models import AgentRunRequest, AgentToolResult
from agent.session_models import AgentTaskRecord, AgentToolTraceRecord, new_id
from agent.session_store import AgentSessionStore
from api.main import app
from schemas.common_schema import ServiceError


def _tool_result(tool_name: str, success: bool, data=None, summary: str = "", error: ServiceError | None = None):
    return AgentToolResult(
        tool_name=tool_name,
        success=success,
        data=data,
        summary=summary,
        error=error,
    )


class AgentSessionStoreTestCase(unittest.TestCase):
    def test_store_can_create_and_query_session_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AgentSessionStore(temp_dir)
            session = store.create_session(title="test")
            store.append_message(session.session_id, "user", "hello")
            task = AgentTaskRecord(
                task_id=new_id(),
                session_id=session.session_id,
                user_input="hello",
                task_type="unknown",
                success=False,
                conclusion="unknown",
                tool_trace=[
                    AgentToolTraceRecord(
                        trace_id=new_id(),
                        task_id="task-1",
                        session_id=session.session_id,
                        tool_name="qa_chat",
                        success=False,
                        summary="failed",
                        error={"code": "QA_FAILED"},
                    )
                ],
            )
            store.append_task_record(task)

            self.assertIsNotNone(store.get_session(session.session_id))
            self.assertEqual(len(store.list_messages(session.session_id)), 1)
            self.assertEqual(len(store.list_task_records(session.session_id)), 1)
            self.assertIsNotNone(store.get_task_record(task.task_id))
            self.assertEqual(len(store.list_tool_traces(session.session_id)), 1)

    def test_list_sessions_returns_recent_sessions_in_updated_at_desc_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AgentSessionStore(temp_dir)
            with patch("agent.session_store.utc_now_iso", return_value="2026-05-04T10:00:00Z"):
                first = store.create_session(title="first")
            with patch("agent.session_store.utc_now_iso", return_value="2026-05-04T10:05:00Z"):
                second = store.create_session(title="second")
            with patch("agent.session_store.utc_now_iso", return_value="2026-05-04T10:10:00Z"):
                store.touch_session(first.session_id)

            sessions = store.list_sessions(limit=10)

            self.assertEqual([session.session_id for session in sessions[:2]], [first.session_id, second.session_id])
            self.assertEqual(sessions[0].updated_at, "2026-05-04T10:10:00Z")

    def test_update_session_title_updates_title_and_keeps_old_sessions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AgentSessionStore(temp_dir)
            old_session = store.create_session(title="old")
            new_session = store.create_session(title="new")

            with patch("agent.session_store.utc_now_iso", return_value="2026-05-04T11:00:00Z"):
                updated = store.update_session_title(old_session.session_id, "快递退款短信分析")

            self.assertIsNotNone(updated)
            self.assertEqual(updated.title, "快递退款短信分析")
            self.assertEqual(updated.updated_at, "2026-05-04T11:00:00Z")

            listed_ids = [session.session_id for session in store.list_sessions(limit=10)]
            self.assertIn(old_session.session_id, listed_ids)
            self.assertIn(new_session.session_id, listed_ids)


class AgentSessionFlowTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = AgentSessionStore(self.temp_dir.name)
        self.store_patcher = patch("agent.agent_service.get_agent_session_store", return_value=self.store)
        self.route_store_patcher = patch("api.routes.session_routes.get_agent_session_store", return_value=self.store)
        self.store_patcher.start()
        self.route_store_patcher.start()
        self.client = TestClient(app, raise_server_exceptions=False)

    def tearDown(self):
        self.store_patcher.stop()
        self.route_store_patcher.stop()
        self.temp_dir.cleanup()

    @patch("agent.agent_service.qa_chat_tool")
    @patch("agent.agent_service.case_search_tool")
    @patch("agent.agent_service.sms_recognize_tool")
    def test_run_agent_without_session_id_creates_session_and_persists_records(self, mock_sms, mock_case, mock_qa):
        mock_sms.return_value = _tool_result(
            "sms_recognize",
            True,
            data={
                "risk_level": "high",
                "fraud_type": "冒充客服诈骗",
                "confidence": 0.9,
                "evidence": ["包含异常链接"],
                "suggestions": ["不要点击陌生链接"],
            },
            summary="完成短信风险识别",
        )
        mock_case.return_value = _tool_result(
            "case_search",
            True,
            data={"cases": [{"title": "客服退款诈骗", "summary": "诱导转账", "source": "neo4j"}], "total_count": 1},
            summary="检索到 1 条相关案例",
        )
        mock_qa.return_value = _tool_result(
            "qa_chat",
            True,
            data={"answer": "请通过官方客服核实，不要提供验证码。"},
            summary="完成建议生成",
        )

        response = run_agent({"user_input": "帮我看看这段短信是不是诈骗：点击链接退款"})

        self.assertTrue(response.success)
        self.assertIsNotNone(response.session_id)
        self.assertIsNotNone(response.task_id)
        messages = self.store.list_messages(response.session_id)
        tasks = self.store.list_task_records(response.session_id)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[1].role, "assistant")
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task_id, response.task_id)
        self.assertGreaterEqual(len(tasks[0].tool_trace), 3)

    @patch("agent.agent_service.qa_chat_tool")
    @patch("agent.agent_service.case_search_tool")
    @patch("agent.agent_service.sms_recognize_tool")
    def test_run_agent_with_session_id_reuses_same_session_and_follow_up_uses_memory(self, mock_sms, mock_case, mock_qa):
        mock_sms.return_value = _tool_result(
            "sms_recognize",
            True,
            data={
                "risk_level": "high",
                "fraud_type": "冒充客服诈骗",
                "confidence": 0.91,
                "evidence": ["包含退款链接"],
                "suggestions": ["不要点击链接"],
            },
            summary="完成短信风险识别",
        )
        mock_case.return_value = _tool_result(
            "case_search",
            True,
            data={"cases": [], "total_count": 0},
            summary="未检索到直接案例",
        )
        mock_qa.side_effect = [
            _tool_result("qa_chat", True, data={"answer": "初步判断为高风险，请勿点击链接。"}, summary="第一轮建议"),
            _tool_result("qa_chat", True, data={"answer": "立即停止操作，联系官方客服并修改密码。"}, summary="追问建议"),
        ]

        first = run_agent({"user_input": "帮我看看这段短信是不是诈骗：您的账户异常，请点击链接验证"})
        second = run_agent(
            {
                "session_id": first.session_id,
                "user_input": "那我现在应该怎么做？",
                "options": {"return_trace": True, "case_limit": 5, "use_memory": True, "history_limit": 6},
            }
        )

        self.assertEqual(first.session_id, second.session_id)
        self.assertTrue(second.context_used)
        self.assertIsNotNone(second.memory_summary)
        self.assertEqual(second.task_type, "text_risk_analysis")
        self.assertIn("停止操作", second.answer)
        self.assertEqual(len(self.store.list_messages(first.session_id)), 4)
        self.assertEqual(len(self.store.list_task_records(first.session_id)), 2)

    @patch("agent.agent_service.qa_chat_tool")
    @patch("agent.agent_service.case_search_tool")
    @patch("agent.agent_service.sms_recognize_tool")
    def test_remedial_follow_up_after_high_risk_sms(self, mock_sms, mock_case, mock_qa):
        mock_sms.return_value = _tool_result(
            "sms_recognize",
            True,
            data={
                "risk_level": "high",
                "fraud_type": "快递退款诈骗",
                "confidence": 0.97,
                "evidence": ["要求填写银行卡号和验证码", "通过链接诱导提交敏感信息"],
                "suggestions": ["不要点击链接", "不要填写银行卡和验证码"],
            },
            summary="完成短信风险识别",
        )
        mock_case.return_value = _tool_result("case_search", True, data={"cases": [], "total_count": 0}, summary="无结果")
        mock_qa.return_value = _tool_result(
            "qa_chat",
            True,
            data={"answer": "这是典型快递退款诈骗，请立即停止操作。"},
            summary="完成建议生成",
        )

        first = run_agent(
            {
                "user_input": "帮我看看这段短信是不是诈骗：您的快递异常，可申请 300 元赔偿，请点击链接填写银行卡号和验证码。"
            }
        )
        second = run_agent(
            {
                "session_id": first.session_id,
                "user_input": "如果我已经填写了银行卡号和验证码呢？",
                "options": {"return_trace": True, "case_limit": 5, "use_memory": True, "history_limit": 6},
            }
        )

        self.assertTrue(second.context_used)
        self.assertTrue(bool(second.memory_summary))
        self.assertNotEqual(second.risk_level, "low")
        self.assertNotIn("未发现明显高危特征", second.conclusion)
        self.assertTrue(any("银行" in item or "冻结" in item for item in second.suggestions))
        self.assertTrue(any("验证码" in item or "账户" in item for item in second.suggestions))
        self.assertTrue(any("96110" in item or "报警" in item for item in second.suggestions))
        self.assertTrue(any(trace.tool_name == "follow_up_handler" for trace in second.tool_trace))

    @patch("agent.agent_service.qa_chat_tool")
    @patch("agent.agent_service.case_search_tool")
    @patch("agent.agent_service.sms_recognize_tool")
    def test_follow_up_after_transfer(self, mock_sms, mock_case, mock_qa):
        mock_sms.return_value = _tool_result(
            "sms_recognize",
            True,
            data={
                "risk_level": "high",
                "fraud_type": "冒充客服诈骗",
                "confidence": 0.95,
                "evidence": ["要求转账到安全账户"],
                "suggestions": ["不要转账到所谓安全账户"],
            },
            summary="完成短信风险识别",
        )
        mock_case.return_value = _tool_result("case_search", True, data={"cases": [], "total_count": 0}, summary="无结果")
        mock_qa.return_value = _tool_result(
            "qa_chat",
            True,
            data={"answer": "这是高风险转账诈骗。"},
            summary="完成建议生成",
        )

        first = run_agent({"user_input": "帮我看下这是不是诈骗：客服让我把钱转到安全账户验证。"})
        second = run_agent(
            {
                "session_id": first.session_id,
                "user_input": "如果我已经转账了怎么办？",
                "options": {"return_trace": True, "case_limit": 5, "use_memory": True, "history_limit": 6},
            }
        )

        self.assertNotEqual(second.risk_level, "low")
        self.assertTrue(any("报警" in item or "96110" in item for item in second.suggestions))
        self.assertTrue(any("证据" in item or "记录" in item for item in second.suggestions))
        self.assertTrue(any("银行" in item or "支付平台" in item for item in second.suggestions))

    @patch("agent.agent_service.qa_chat_tool")
    @patch("agent.agent_service.case_search_tool")
    @patch("agent.agent_service.sms_recognize_tool")
    def test_explanation_follow_up(self, mock_sms, mock_case, mock_qa):
        mock_sms.return_value = _tool_result(
            "sms_recognize",
            True,
            data={
                "risk_level": "high",
                "fraud_type": "快递退款诈骗",
                "confidence": 0.93,
                "evidence": ["包含异常链接", "索要银行卡和验证码"],
                "suggestions": ["不要点击链接"],
            },
            summary="完成短信风险识别",
        )
        mock_case.return_value = _tool_result("case_search", True, data={"cases": [], "total_count": 0}, summary="无结果")
        mock_qa.side_effect = [
            _tool_result("qa_chat", True, data={"answer": "这是高风险诈骗短信。"}, summary="首轮建议"),
            _tool_result(
                "qa_chat",
                True,
                data={"answer": "之所以判断为快递退款诈骗，是因为短信包含异常链接，并索要银行卡和验证码。"},
                summary="解释型追问",
            ),
        ]

        first = run_agent({"user_input": "帮我看看这段短信是不是诈骗：您的快递异常，请点击链接填写银行卡号和验证码。"})
        second = run_agent(
            {
                "session_id": first.session_id,
                "user_input": "为什么这是诈骗？",
                "options": {"return_trace": True, "case_limit": 5, "use_memory": True, "history_limit": 6},
            }
        )

        self.assertTrue(second.context_used)
        self.assertTrue("快递退款诈骗" in (second.answer or "") or "包含异常链接" in (second.answer or ""))
        self.assertEqual(second.task_type, "text_risk_analysis")

    def test_unknown_without_memory_still_unknown(self):
        response = run_agent({"user_input": "如果我已经填了怎么办？"})
        self.assertFalse(response.success)
        self.assertEqual(response.task_type, "unknown")

    def test_out_of_scope_task_is_recorded(self):
        response = run_agent({"user_input": "你好呀"})
        self.assertTrue(response.success)
        self.assertEqual(response.task_type, "out_of_scope")
        tasks = self.store.list_task_records(response.session_id)
        self.assertEqual(tasks[-1].task_type, "out_of_scope")
        self.assertEqual(tasks[-1].metadata["handler_name"], "out_of_scope_handler")
        self.assertEqual(tasks[-1].metadata["error_type"], "OUT_OF_SCOPE")

    @patch("agent.agent_service.ocr_tool")
    def test_ocr_fallback_is_recorded(self, mock_ocr):
        mock_ocr.return_value = _tool_result(
            "ocr",
            False,
            error=ServiceError(code="OCR_FAILED", message="OCR unavailable"),
            summary="OCR failed",
        )

        response = run_agent(AgentRunRequest(user_input="帮我分析这张截图", input_type="image", image="fake-image"))

        self.assertFalse(response.success)
        self.assertEqual(response.error.code, "OCR_FAILED")
        task = self.store.list_task_records(response.session_id)[-1]
        self.assertEqual(task.error["code"], "OCR_FAILED")
        self.assertIsNotNone(task.fallback_message)
        self.assertEqual(task.tool_trace[0].error["code"], "OCR_FAILED")

    @patch("agent.agent_service.qa_chat_tool")
    @patch("agent.agent_service.case_search_tool")
    @patch("agent.agent_service.sms_recognize_tool")
    def test_agent_session_api_returns_messages_and_tasks(self, mock_sms, mock_case, mock_qa):
        mock_sms.return_value = _tool_result(
            "sms_recognize",
            True,
            data={
                "risk_level": "high",
                "fraud_type": "冒充客服诈骗",
                "confidence": 0.9,
                "evidence": ["包含异常链接"],
                "suggestions": ["不要点击陌生链接"],
            },
            summary="完成短信风险识别",
        )
        mock_case.return_value = _tool_result("case_search", True, data={"cases": [], "total_count": 0}, summary="无结果")
        mock_qa.return_value = _tool_result("qa_chat", True, data={"answer": "不要转账。"}, summary="完成建议")

        run_response = self.client.post(
            "/agent/run",
            json={
                "user_input": "帮我看看这段短信是不是诈骗：点击链接退款",
                "input_type": "text",
                "options": {"return_trace": True, "case_limit": 5, "use_memory": True, "history_limit": 6},
            },
        )
        body = run_response.json()
        session_id = body["session_id"]

        messages_resp = self.client.get(f"/agent/sessions/{session_id}/messages", params={"limit": 10})
        tasks_resp = self.client.get(f"/agent/sessions/{session_id}/tasks", params={"limit": 10})

        self.assertEqual(messages_resp.status_code, 200)
        self.assertEqual(tasks_resp.status_code, 200)
        self.assertEqual(len(messages_resp.json()["messages"]), 2)
        self.assertEqual(len(tasks_resp.json()["tasks"]), 1)

    def test_create_session_api(self):
        response = self.client.post("/agent/sessions", json={"title": "帮爸妈分析诈骗风险", "metadata": {"source": "ui"}})
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(body["success"])
        self.assertIn("session_id", body)
        self.assertEqual(body["session"]["title"], "帮爸妈分析诈骗风险")

    def test_list_sessions_api_returns_recent_sessions(self):
        with patch("agent.session_store.utc_now_iso", return_value="2026-05-06T10:00:00Z"):
            older = self.store.create_session(title="older")
        with patch("agent.session_store.utc_now_iso", return_value="2026-05-06T10:05:00Z"):
            newer = self.store.create_session(title="newer")
        with patch("agent.session_store.utc_now_iso", return_value="2026-05-06T12:00:00Z"):
            self.store.touch_session(newer.session_id)

        response = self.client.get("/agent/sessions", params={"limit": 20})
        body = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(body["success"])
        self.assertEqual(len(body["sessions"]), 2)
        self.assertEqual(body["sessions"][0]["session_id"], newer.session_id)
        self.assertEqual(body["sessions"][1]["session_id"], older.session_id)


if __name__ == "__main__":
    unittest.main()
