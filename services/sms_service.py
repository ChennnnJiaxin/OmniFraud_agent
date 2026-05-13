from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import jieba

from infra.config import AppConfig
from schemas.common_schema import ServiceError
from schemas.sms_schema import SmsRecognizeResponse

KEYWORDS_PATH = Path("recognize/fraud_keywords.json")


@lru_cache(maxsize=1)
def load_keywords() -> list[str]:
    with KEYWORDS_PATH.open("r", encoding="utf-8") as file:
        keywords = json.load(file)
    return [item[0] for item in keywords]


@lru_cache(maxsize=1)
def load_msg_cls_model():
    from recognize import fraud_msg_cls

    return fraud_msg_cls.MsgClsModel()


def extract_keywords(text: str, top_k: int = 3) -> list[str]:
    words = jieba.lcut(text)
    stopwords = {"的", "了", "是", "在", "和", "就", "都", "而", "可", "中", "这", "那", "有"}
    keywords = set(load_keywords())
    danger_words = [word for word in words if len(word) > 1 and word not in stopwords and word in keywords]
    word_counts = Counter(danger_words)
    result = [word for word, _ in word_counts.most_common(top_k)]
    return result or ["无"]


def get_risk_level(prediction: str, probability: float) -> str:
    if prediction == "无风险":
        return "无风险"
    if probability > 0.7:
        return "高风险"
    if probability > 0.5:
        return "中风险"
    return "低风险"


def _calculate_link_risk(text: str) -> int:
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.[^\s]+"
    urls = re.findall(url_pattern, text)
    return min(len(urls) * 60 + 30, 100)


def _calculate_keyword_risk(text: str) -> int:
    words = jieba.lcut(text)
    hit_count = sum(1 for word in words if word in set(load_keywords()))
    return min(hit_count * 20 + 28, 100)


def _calculate_urgency_score(text: str) -> int:
    urgency_words = ["立即", "马上", "尽快", "赶快", "今天", "现在", "机会"]
    words = jieba.lcut(text)
    count = sum(1 for word in words if word in urgency_words)
    return min(count * 25 + 32, 100)


def predict_text(text: str) -> dict[str, object]:
    predictions = load_msg_cls_model().predict(text)
    max_category, max_prob = predictions[0]
    return {
        "prediction": max_category,
        "probability": float(max_prob),
        "features": {
            "风险等级": get_risk_level(max_category, float(max_prob)),
            "关键词": extract_keywords(text),
            "关键词风险": _calculate_keyword_risk(text),
            "链接风险": _calculate_link_risk(text),
            "紧迫性指数": _calculate_urgency_score(text),
            "语义异常度": float(max_prob) * 100,
        },
        "full_predictions": predictions,
    }


def _default_suggestions(risk_level: str) -> list[str]:
    if risk_level == "无风险":
        return ["保持警惕，不轻信陌生链接和转账要求。"]
    if risk_level == "低风险":
        return ["先核验发送方身份，再决定是否继续操作。", "不要点击短信中的陌生链接。"]
    if risk_level == "中风险":
        return ["暂停任何转账和验证码操作。", "通过官方渠道联系平台或银行核实。"]
    return ["不要点击链接或转账。", "保留证据并通过官方渠道核验。", "如已受骗请尽快报警并联系银行止付。"]


def recognize_sms(text: str) -> SmsRecognizeResponse:
    normalized_text = (text or "").strip()
    if not normalized_text:
        return SmsRecognizeResponse(
            success=False,
            error=ServiceError(code="EMPTY_TEXT", message="请输入要识别的短信或文本内容。"),
        )
    if len(normalized_text) < 10:
        return SmsRecognizeResponse(
            success=False,
            error=ServiceError(code="TEXT_TOO_SHORT", message="输入文本过短，请至少输入10个字符。"),
        )

    try:
        result = predict_text(normalized_text)
        risk_level = str(result["features"]["风险等级"])
        keywords = [str(item) for item in result["features"]["关键词"]]
        evidence = [f"命中关键词：{', '.join(keywords)}"]
        evidence.append(f"链接风险分：{result['features']['链接风险']}")
        evidence.append(f"紧迫性指数：{result['features']['紧迫性指数']}")
        fraud_type = None if result["prediction"] == "无风险" else str(result["prediction"])
        return SmsRecognizeResponse(
            success=True,
            risk_level=risk_level,
            fraud_type=fraud_type,
            confidence=float(result["probability"]),
            evidence=evidence,
            suggestions=_default_suggestions(risk_level),
            raw_result=result,
        )
    except Exception as exc:
        return SmsRecognizeResponse(
            success=False,
            error=ServiceError(code="SMS_RECOGNIZE_FAILED", message="短信识别失败。", detail={"error": str(exc)}),
        )


def generate_sms_suggestions_stream(text: str, prediction: str, config: AppConfig | None = None):
    from clients.llm_client import LlmClient

    prompt = f"""
我这里有一条疑似诈骗的信息，以下是信息内容：
{text}

模型预测风险等级为：
{prediction}

请给出一份约200字、条理清晰、适合普通用户阅读的防范建议。
不要输出概率，适量加入 emoji，但不要过度。
"""
    return LlmClient(config).stream_chat(
        model=None,
        messages=[
            {"role": "system", "content": "你是一名反诈宣传助手。"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=1.0,
    )
