import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger

@OPERATOR_REGISTRY.register()
class PairQualSampleEvaluator:
    """
    单样本/批量文本质量打分器：
    - 加载 zks2856/PairQual-Scorer-en 或 zks2856/PairQual-Scorer-zh
    - 输出经 sigmoid 的分数，范围 [0, 1]
    - 不继承 OperatorABC（避免抽象类 run 冲突）
    """

    def __init__(
        self,
        model_cache_dir: str = "./dataflow_cache",
        device: str | None = None,         # 'cpu' / 'cuda' / None(自动)
        lang: str = "en",                  # 'en' or 'zh'
        max_length: int = 512,
    ):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__}...")

        # 设备处理与回退
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            if device == "cuda" and not torch.cuda.is_available():
                self.logger.warning("CUDA requested but not available; falling back to CPU.")
                self.device = "cpu"
            else:
                self.device = device

        self.model_cache_dir = model_cache_dir
        self.lang = lang
        self.max_length = max_length
        self.score_name = "PairQualScore"

        if self.lang not in ("en", "zh"):
            raise ValueError("Invalid 'lang'. Only 'en' or 'zh' are allowed.")

        model_name = "zks2856/PairQual-Scorer-en" if self.lang == "en" else "zks2856/PairQual-Scorer-zh"

        # 直接用 Auto* 加载仓库自带回归头，避免线性层维度 mismatch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=self.model_cache_dir
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=self.model_cache_dir
        ).to(self.device).eval()

        self.logger.info(f"{self.__class__.__name__} initialized on device: {self.device}")

    def _score_batch(self, texts: list[str]) -> np.ndarray:
        """对一批文本打分，返回 [0,1] 数组"""
        if not texts:
            return np.zeros((0,), dtype=np.float32)

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            if not hasattr(outputs, "logits"):
                raise RuntimeError("Model outputs do not contain 'logits'.")
            logits = outputs.logits.squeeze(-1)  # [B] 或 [B,1] -> [B]
            scores = torch.sigmoid(logits)       # 归一化到 [0,1]

        return scores.detach().cpu().numpy().astype(np.float32)

    def eval(self, dataframe, input_key: str, batch_size: int = 16) -> np.ndarray:
        """对 DataFrame[input_key] 批量打分，返回 np.ndarray，范围 [0,1]"""
        texts = dataframe[input_key].tolist()
        all_scores: list[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_scores = self._score_batch(batch)
            all_scores.append(batch_scores)

        if not all_scores:
            return np.zeros((0,), dtype=np.float32)

        scores = np.concatenate(all_scores, axis=0)
        # 保险裁剪，确保在 [0,1]
        return np.clip(scores, 0.0, 1.0)
