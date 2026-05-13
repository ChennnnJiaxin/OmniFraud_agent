import tempfile
import unittest
from unittest.mock import patch

from agent.agent_service import run_agent
from agent.session_store import AgentSessionStore


class AgentFollowUpProcedureTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = AgentSessionStore(self.temp_dir.name)
        self.store_patcher = patch("agent.agent_service.get_agent_session_store", return_value=self.store)
        self.store_patcher.start()

    def tearDown(self):
        self.store_patcher.stop()
        self.temp_dir.cleanup()

    def test_alarm_phone_follow_up_after_remedial_action(self):
        first = run_agent({"user_input": "但是我已经给了验证码诶"})
        second = run_agent(
            {
                "session_id": first.session_id,
                "user_input": "那报警电话是什么",
                "options": {"return_trace": True, "case_limit": 5, "use_memory": True, "history_limit": 6},
            }
        )

        combined = "\n".join([second.conclusion or "", second.answer or "", " ".join(second.suggestions or [])])
        self.assertNotEqual(second.task_type, "unknown")
        self.assertTrue(second.context_used)
        self.assertIn("110", combined)
        self.assertIn("96110", combined)
        self.assertTrue("银行" in combined or "支付平台" in combined)
        task = self.store.list_task_records(second.session_id)[-1]
        self.assertEqual(task.metadata["follow_up_type"], "procedure_question")

    def test_96110_follow_up_without_history(self):
        response = run_agent({"user_input": "96110是什么"})

        combined = "\n".join([response.conclusion or "", response.answer or "", " ".join(response.suggestions or [])])
        self.assertNotEqual(response.task_type, "unknown")
        self.assertIn("反诈", combined)
        self.assertIn("96110", combined)
        self.assertTrue("110" in combined or "报警" in combined)

    def test_evidence_follow_up_after_emergency_loss(self):
        first = run_agent({"user_input": "我已经被骗了100万了"})
        second = run_agent(
            {
                "session_id": first.session_id,
                "user_input": "我要保留什么证据",
                "options": {"return_trace": True, "case_limit": 5, "use_memory": True, "history_limit": 6},
            }
        )

        combined = "\n".join([second.conclusion or "", second.answer or "", " ".join(second.suggestions or [])])
        self.assertNotEqual(second.task_type, "unknown")
        self.assertTrue(second.context_used)
        self.assertIn("聊天记录", combined)
        self.assertIn("转账记录", combined)
        self.assertTrue(any(keyword in combined for keyword in ("手机号", "账号", "链接")))

    def test_bank_freeze_follow_up_after_code_leak(self):
        first = run_agent({"user_input": "我已经把验证码告诉他了"})
        second = run_agent(
            {
                "session_id": first.session_id,
                "user_input": "怎么冻结银行卡",
                "options": {"return_trace": True, "case_limit": 5, "use_memory": True, "history_limit": 6},
            }
        )

        combined = "\n".join([second.conclusion or "", second.answer or "", " ".join(second.suggestions or [])])
        self.assertNotEqual(second.task_type, "unknown")
        self.assertIn("银行", combined)
        self.assertTrue(any(keyword in combined for keyword in ("冻结", "挂失", "止付")))


if __name__ == "__main__":
    unittest.main()
