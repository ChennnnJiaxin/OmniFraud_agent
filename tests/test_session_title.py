import unittest
from types import SimpleNamespace

from agent.session_models import AgentMessage
from agent.session_title import fallback_session_title, generate_session_title, is_default_session_title


def _message(role: str, content: str) -> AgentMessage:
    return AgentMessage(
        message_id=f"{role}-1",
        session_id="session-1",
        role=role,
        content=content,
        created_at="2026-05-06T12:00:00Z",
    )


class FakeLlmClient:
    def __init__(self, content: str):
        self.content = content

    def complete(self, **kwargs):
        del kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
        )


class FailingLlmClient:
    def complete(self, **kwargs):
        del kwargs
        raise RuntimeError("title generation failed")


class SessionTitleTestCase(unittest.TestCase):
    def test_is_default_session_title(self):
        self.assertTrue(is_default_session_title(None, "新的反诈会话"))
        self.assertTrue(is_default_session_title("新的反诈会话", "新的反诈会话"))
        self.assertFalse(is_default_session_title("被骗转账后如何止损", "新的反诈会话"))

    def test_fallback_session_title_uses_first_user_message(self):
        title = fallback_session_title(
            [
                _message("user", "验证码泄露以后应该怎么补救，需要先改哪些账号"),
                _message("assistant", "先停用高风险账号，并联系银行核验。"),
            ],
            "新的反诈会话",
        )
        self.assertEqual(title, "验证码泄露以后应该怎么补救需要先改哪")

    def test_generate_session_title_uses_llm_output_and_normalizes(self):
        title = generate_session_title(
            [
                _message("user", "我被冒充客服骗着点开了退款链接，现在担心银行卡被盗刷"),
                _message("assistant", "先停止继续操作，联系银行客服核验并修改相关密码。"),
            ],
            default_title="新的反诈会话",
            llm_client=FakeLlmClient("标题：被骗转账后如何止损。"),
        )
        self.assertEqual(title, "被骗转账后如何止损")

    def test_generate_session_title_falls_back_when_llm_fails(self):
        title = generate_session_title(
            [
                _message("user", "父母经常在微信和抖音上看陌生直播，这类场景要怎么做防骗画像"),
                _message("assistant", "可以从常用平台、付款习惯和高频触发场景来评估风险。"),
            ],
            default_title="新的反诈会话",
            llm_client=FailingLlmClient(),
        )
        self.assertEqual(title, "父母经常在微信和抖音上看陌生直播这类")


if __name__ == "__main__":
    unittest.main()
