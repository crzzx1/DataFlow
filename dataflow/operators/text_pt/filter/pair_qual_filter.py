from dataflow.operators.text_pt import PairQualSampleEvaluator
import numpy as np
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class PairQualFilter(OperatorABC):
    """
    DataFlow Operator：
    - 从 storage 读 DataFrame
    - 对 input_key 列打分（0–1）
    - 写回 score_key 列并按区间过滤
    """

    def __init__(
        self,
        min_score: float = 0.0,           # 建议默认阈值
        max_score: float = 1.0,           # Score 区间 [0,1]
        model_cache_dir: str = "./dataflow_cache",
        lang: str = "en",
        device: str | None = None,        # "cpu"/"cuda"/None
        input_key: str = "raw_content",
        score_key: str = "PairQualScore",
        batch_size: int = 8,
    ):
        self.logger = get_logger()

        if not (0.0 <= min_score <= 1.0 and 0.0 <= max_score <= 1.0 and min_score <= max_score):
            raise ValueError(
                f"PairQualFilter expects scores in [0,1]. Got min_score={min_score}, max_score={max_score}."
            )

        self.min_score = float(min_score)
        self.max_score = float(max_score)
        self.input_key = input_key
        self.score_key = score_key
        self.batch_size = int(batch_size)

        # 初始化评估器（工具类） ——— 注意缩进！
        self.scorer = PairQualSampleEvaluator(
            model_cache_dir=model_cache_dir,
            lang=lang,
            device=device,
        )

        self.filter_name = "PairQualFilter"
        self.logger.info(
            f"Initializing {self.filter_name} with min_score={self.min_score}, max_score={self.max_score}, "
            f"input_key='{self.input_key}', score_key='{self.score_key}'."
        )

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于 PairQualScorer 的文本质量过滤算子，得分区间为 [0,1]，越高越好。\n"
                "参数：min_score/max_score（区间内保留），model_cache_dir，lang（en/zh），device（cpu/cuda/None），"
                "input_key（默认 raw_content），score_key（默认 PairQualScore）。"
            )
        else:
            return (
                "PairQual-based quality filtering operator. Scores in [0,1], higher is better.\n"
                "Params: min_score/max_score (keep inside range), model_cache_dir, lang (en/zh), "
                "device (cpu/cuda/None), input_key (default raw_content), score_key (default PairQualScore)."
            )

    def run(self, storage: DataFlowStorage, input_key: str | None = None):
        # 读取数据
        df = storage.read("dataframe")
        if df is None or df.empty:
            self.logger.warning(f"{self.filter_name}: empty dataframe, skip.")
            return [self.score_key]

        col = input_key or self.input_key
        if col not in df.columns:
            self.logger.warning(f"{self.filter_name}: input_key '{col}' not found, skip.")
            return [self.score_key]

        # 打分（0–1）
        try:
            # 优先尝试带 batch_size（若 eval 支持）
            scores = self.scorer.eval(df, col, batch_size=self.batch_size)
        except TypeError:
            # 兼容旧版不支持 batch_size 的实现
            scores = self.scorer.eval(df, col)

        scores = np.clip(scores, 0.0, 1.0)  # 保险裁剪
        df[self.score_key] = scores

        # 区间过滤
        mask = (df[self.score_key] >= self.min_score) & (df[self.score_key] <= self.max_score)
        kept = int(mask.sum())
        self.logger.info(f"{self.filter_name}: keep {kept}/{len(df)} rows in [{self.min_score}, {self.max_score}].")

        df = df[mask].reset_index(drop=True)
        storage.write(df)
        return [self.score_key]
