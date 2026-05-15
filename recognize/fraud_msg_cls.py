# %%
import os
from pathlib import Path

import torch
from sklearn.preprocessing import LabelEncoder
from transformers import BertForSequenceClassification, BertTokenizer

classes = [
    "冒充公检法及政府机关类",
    "冒充军警购物类诈骗",
    "冒充电商物流客服类",
    "冒充领导、熟人类",
    "无风险",
    "网络婚恋、交友类",
    "网黑案件",
    "虚假信用服务类",
    "虚假网络投资理财类",
    "虚假购物、服务类",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-chinese"
MODEL_WEIGHTS_PATH = Path("model/final_model.pth")


def _local_files_only() -> bool:
    return os.getenv("TRANSFORMERS_OFFLINE", "").strip().lower() in {"1", "true", "yes"}


def _load_base_tokenizer():
    try:
        return BertTokenizer.from_pretrained(MODEL_NAME, local_files_only=_local_files_only())
    except Exception as exc:
        raise RuntimeError(
            f"无法加载 {MODEL_NAME} 的 tokenizer。云端部署会尝试联网下载；"
            f"如需离线运行，请先下载缓存并设置 TRANSFORMERS_OFFLINE=1。原始错误：{exc}"
        ) from exc


def _load_base_model(num_labels: int):
    try:
        return BertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
            local_files_only=_local_files_only(),
        ).to(device)
    except Exception as exc:
        raise RuntimeError(
            f"无法加载 {MODEL_NAME} 的基础模型。云端部署会尝试联网下载；"
            f"如需离线运行，请先下载缓存并设置 TRANSFORMERS_OFFLINE=1。原始错误：{exc}"
        ) from exc


def _load_state_dict(path: Path):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class MsgClsModel:
    def __init__(self):
        self.lb = LabelEncoder()
        self.lb.classes_ = classes
        self.tokenizer = _load_base_tokenizer()
        self.model = _load_base_model(len(classes))

        if not MODEL_WEIGHTS_PATH.exists():
            print("Model not found, downloading...")
            import requests

            url = "https://dlink.host/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvYy82YWE4YmQ4MzYxZWQ0NTkwL0VYT1F5WktPTnJSUG1vdENncVVodVI4QjZ4MV9uMjVfMDhFTGdlOHNURF9Fcnc_ZT05ZXB4WE4.pth"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            MODEL_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with MODEL_WEIGHTS_PATH.open("wb") as f:
                f.write(response.content)

        self.model.load_state_dict(_load_state_dict(MODEL_WEIGHTS_PATH))
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def encode_texts(self, texts):
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

    def predict(self, test_text):
        output = self.model(**self.encode_texts([test_text]).to(device))
        output = output.logits
        output = torch.softmax(output, dim=1)
        output = output.cpu().detach().numpy()
        output = output[0]
        output = list(zip(self.lb.classes_, output))
        return sorted(output, key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    model = MsgClsModel()
    test_text = "【顺丰】尊敬的客户，您使用顺丰的频率较高，现赠送您暖风扇一只，请添加支付宝好友进行登记领取。"
    print(model.predict(test_text))
