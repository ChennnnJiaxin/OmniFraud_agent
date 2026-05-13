import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.agent_service import run_agent
from agent.models import AgentToolResult
from agent.session_store import AgentSessionStore


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "agent_eval_cases.json"
RISK_ORDER = {"unknown": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


class AgentEvalRegressionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = AgentSessionStore(self.temp_dir.name)
        self.store_patcher = patch("agent.agent_service.get_agent_session_store", return_value=self.store)
        self.sms_patcher = patch("agent.agent_service.sms_recognize_tool", side_effect=self._fake_sms_tool)
        self.case_patcher = patch("agent.agent_service.case_search_tool", side_effect=self._fake_case_search_tool)
        self.qa_patcher = patch("agent.agent_service.qa_chat_tool", side_effect=self._fake_qa_tool)
        self.risk_patcher = patch("agent.agent_service.risk_report_tool", side_effect=self._fake_risk_report_tool)
        self.graph_patcher = patch("agent.agent_service.graph_query_tool", side_effect=self._fake_graph_tool)
        self.ocr_patcher = patch("agent.agent_service.ocr_tool", side_effect=self._fake_ocr_tool)
        for patcher in (
            self.store_patcher,
            self.sms_patcher,
            self.case_patcher,
            self.qa_patcher,
            self.risk_patcher,
            self.graph_patcher,
            self.ocr_patcher,
        ):
            patcher.start()

    def tearDown(self):
        for patcher in (
            self.ocr_patcher,
            self.graph_patcher,
            self.risk_patcher,
            self.qa_patcher,
            self.case_patcher,
            self.sms_patcher,
            self.store_patcher,
        ):
            patcher.stop()
        self.temp_dir.cleanup()

    def test_agent_eval_regression_cases(self):
        failures: list[str] = []
        for case in self.cases:
            try:
                self._assert_case(case)
            except AssertionError as exc:
                failures.append(f"{case['id']}: {exc}")
        if failures:
            self.fail("回归样本失败：\n" + "\n".join(failures))

    def _assert_case(self, case: dict):
        session_id = None
        for preamble in case.get("session_preamble", []):
            preamble_response = run_agent({"session_id": session_id, "user_input": preamble})
            session_id = preamble_response.session_id

        response = run_agent({"session_id": session_id, "user_input": case["user_input"]})

        expected_success = case.get("expected_success", True)
        self.assertEqual(response.success, expected_success, "success 不符合预期")
        if "expected_task_type" in case:
            self.assertEqual(response.task_type, case["expected_task_type"], "task_type 不符合预期")
        self.assertEqual(response.safety_intent, case["expected_safety_intent"], "safety_intent 不符合预期")

        expected_min_risk_level = case.get("expected_min_risk_level")
        if expected_min_risk_level:
            self.assertGreaterEqual(
                RISK_ORDER.get(response.risk_level, 0),
                RISK_ORDER[expected_min_risk_level],
                f"risk_level={response.risk_level} 低于预期 {expected_min_risk_level}",
            )

        if case["category"] in {"remedial_action", "emergency_loss", "secondary_fraud"}:
            self.assertNotEqual(response.task_type, "unknown", "高危样本不允许返回 unknown")

        combined_text = "\n".join(
            [
                response.conclusion or "",
                response.answer or "",
                " ".join(response.suggestions or []),
            ]
        )
        for keyword in case.get("must_include", []):
            self.assertIn(keyword, combined_text, f"缺少关键提示词：{keyword}")
        for keyword in case.get("must_not_include", []):
            self.assertNotIn(keyword, combined_text, f"出现了禁止词：{keyword}")

        task = self.store.list_task_records(response.session_id)[-1]
        for key in (
            "safety_intent",
            "handler_name",
            "context_used",
            "prompt_version",
            "response_policy_version",
            "duration_ms",
        ):
            self.assertIn(key, task.metadata, f"task_record.metadata 缺少 {key}")

    def _fake_sms_tool(self, text: str) -> AgentToolResult:
        text = text or ""
        fraud_type = "冒充客服诈骗"
        if any(keyword in text for keyword in ("刷单", "垫付", "返利")):
            fraud_type = "刷单返利诈骗"
        elif any(keyword in text for keyword in ("安全账户", "公检法", "洗钱")):
            fraud_type = "冒充公检法诈骗"
        elif any(keyword in text for keyword in ("投资", "充值", "平台", "荐股")):
            fraud_type = "虚假投资理财诈骗"
        elif any(keyword in text for keyword in ("快递", "退款", "QQ")):
            fraud_type = "冒充客服退款诈骗"

        return AgentToolResult(
            tool_name="sms_recognize",
            success=True,
            data={
                "risk_level": "high",
                "fraud_type": fraud_type,
                "confidence": 0.93,
                "evidence": ["包含高风险诈骗话术", "存在诱导点击链接或转账特征"],
                "suggestions": ["不要点击陌生链接", "不要提供验证码", "通过官方渠道核实"],
            },
            summary="完成短信风险识别",
            duration_ms=5,
        )

    def _fake_case_search_tool(self, query: str, fraud_type: str | None = None, limit: int = 5) -> AgentToolResult:
        case_type = fraud_type or query or "诈骗"
        return AgentToolResult(
            tool_name="case_search",
            success=True,
            data={
                "cases": [
                    {
                        "title": f"{case_type}案例",
                        "summary": f"{case_type}常见套路包括伪造身份、催促操作和诱导转账。",
                        "source": "mock",
                    }
                ],
                "total_count": 1,
            },
            summary="检索到 1 条相关案例",
            duration_ms=4,
        )

    def _fake_qa_tool(self, question: str, context=None) -> AgentToolResult:
        answer = "请通过官方渠道核实，不要点击链接、不要提供验证码，也不要继续转账。"
        if "为什么" in question:
            answer = "之所以判断为诈骗，是因为内容中出现了伪造身份、催促操作和诱导转账等典型风险信号。"
        elif any(keyword in question for keyword in ("怎么办", "下一步", "能不能点", "继续和他说")):
            answer = "建议立刻停止按对方要求操作，通过官方渠道核实，并保留聊天与转账记录。"
        elif "案例" in question or "总结" in question:
            answer = "这类案例通常先制造信任或紧迫感，再诱导点击链接、下载 App 或转账。"
        elif "画像" in question or "人群" in question or "容易遇到什么骗局" in question:
            answer = "这类人群更容易被高回报承诺、情感关怀或官方身份话术诱导，需要重点提醒其只走官方渠道。"
        return AgentToolResult(
            tool_name="qa_chat",
            success=True,
            data={"answer": answer},
            summary="问答服务生成了补充建议",
            duration_ms=6,
        )

    def _fake_risk_report_tool(self, profile: dict) -> AgentToolResult:
        text = " ".join(str(item) for item in profile.values() if item)
        risk_level = "medium"
        vulnerable = ["冒充客服诈骗"]
        if any(keyword in text for keyword in ("大学生", "学生", "刷单")):
            risk_level = "high"
            vulnerable = ["刷单返利诈骗"]
        elif any(keyword in text for keyword in ("养生", "老人", "60", "保健品")):
            risk_level = "medium"
            vulnerable = ["保健品诈骗", "冒充客服诈骗"]
        elif any(keyword in text for keyword in ("游戏", "充值")):
            risk_level = "medium"
            vulnerable = ["游戏充值诈骗"]
        return AgentToolResult(
            tool_name="risk_report",
            success=True,
            data={
                "risk_level": risk_level,
                "vulnerable_fraud_types": vulnerable,
                "reasons": ["容易受高收益或情绪化话术影响"],
                "suggestions": ["只通过官方渠道核实，不轻信高收益或紧急要求"],
                "report": "请重点防范社交和支付场景中的诈骗诱导。",
            },
            summary="完成人群风险评估",
            duration_ms=4,
        )

    def _fake_graph_tool(self, query: str) -> AgentToolResult:
        return AgentToolResult(
            tool_name="graph_query",
            success=True,
            data={"explanation": f"{query} 常见链路是建立信任、制造紧迫感、诱导转账。"},
            summary="完成图谱查询",
            duration_ms=3,
        )

    def _fake_ocr_tool(self, image_input) -> AgentToolResult:
        return AgentToolResult(
            tool_name="ocr",
            success=True,
            data={"text": "您的账户异常，请点击链接验证身份"},
            summary="OCR 成功提取图片文本",
            duration_ms=5,
        )


if __name__ == "__main__":
    unittest.main()
