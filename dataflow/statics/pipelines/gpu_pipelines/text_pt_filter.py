from dataflow.operators.general_text import (
    BlocklistFilter,
    CapitalWordsFilter,
    CharNumberFilter,
    ColonEndFilter,
    ContentNullFilter,
    CurlyBracketFilter,
    IDCardFilter,
    HtmlEntityFilter,
    HtmlUrlRemoverRefiner,
    LanguageFilter,
    LineEndWithEllipsisFilter,
    LineStartWithBulletpointFilter,
    LineWithJavascriptFilter,
    LoremIpsumFilter,
    MeanWordLengthFilter,
    MinHashDeduplicateFilter,
    NoPuncFilter,
    RemoveEmojiRefiner,
    RemoveExtraSpacesRefiner,
    SentenceNumberFilter,
    SpecialCharacterFilter,
    SymbolWordRatioFilter,
    UniqueWordsFilter,
    WatermarkFilter,
    WordNumberFilter,
)
from dataflow.operators.text_pt import PairQualFilter
from dataflow.utils.storage import FileStorage


class PTTextFilter_GPUPipeline:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="/root/DataFlow/pt_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        self.model_cache_dir = "./dataflow_cache"
        self.language_filter = LanguageFilter(
            allowed_languages="__label__eng_Latn",
            model_cache_dir=self.model_cache_dir,
        )

        # --- Refiners & Filters ---
        self.remove_extra_spaces_refiner = RemoveExtraSpacesRefiner()
        self.remove_emoji_refiner = RemoveEmojiRefiner()
        self.html_remove_refiner = HtmlUrlRemoverRefiner()

        self.minhash_deduplicator = MinHashDeduplicateFilter(
            num_perm=128, threshold=0.9, use_n_gram=True, ngram=5
        )
        self.blocklist_filter = BlocklistFilter()
        self.word_number_filter = WordNumberFilter(min_words=5, max_words=100000)

        self.colon_end_filter = ColonEndFilter()
        self.sentence_number_filter = SentenceNumberFilter(min_sentences=1, max_sentences=7500)
        self.line_end_with_ellipsis_filter = LineEndWithEllipsisFilter(threshold=0.3)

        self.content_null_filter = ContentNullFilter()
        self.mean_word_length_filter = MeanWordLengthFilter(min_length=3, max_length=10)
        self.symbol_word_ratio_filter = SymbolWordRatioFilter(threshold=0.4)
        self.html_entity_filter = HtmlEntityFilter()
        self.id_card_filter = IDCardFilter(threshold=3)
        self.no_punc_filter = NoPuncFilter(threshold=112)
        self.special_character_filter = SpecialCharacterFilter()

        self.watermark_filter = WatermarkFilter(
            watermarks=["Copyright", "Watermark", "Confidential"]
        )
        self.curly_bracket_filter = CurlyBracketFilter(threshold=0.025)
        self.capital_words_filter = CapitalWordsFilter(threshold=0.2, use_tokenizer=False)
        self.lorem_ipsum_filter = LoremIpsumFilter(threshold=3e-8)
        self.unique_words_filter = UniqueWordsFilter(threshold=0.1)
        self.char_number_filter = CharNumberFilter(threshold=100)

        self.line_start_with_bulletpoint_filter = LineStartWithBulletpointFilter(threshold=0.9)
        self.line_with_javascript_filter = LineWithJavascriptFilter(threshold=3)

        # ✅ 修正：PairQual 设为 0–1 区间
        self.pairqual_kwargs = dict(min_score=0.0, max_score=1.0, lang="en", device="cpu")
        self.quality_filter = None

    def _run_operator(self, operator, input_key: str = "raw_content") -> None:
        operator.run(storage=self.storage.step(), input_key=input_key)

    # ✅ 修正：Stage 1 先清洗再语言识别，降低 LID 误判率
    def _stage_language_and_initial_cleaning(self) -> None:
        for operator in (
            self.remove_extra_spaces_refiner,
            self.remove_emoji_refiner,
            self.html_remove_refiner,
            self.language_filter,  # moved to after cleaning
        ):
            self._run_operator(operator)

    def _stage_deduplication_and_basic_filters(self) -> None:
        for operator in (
            self.minhash_deduplicator,
            self.blocklist_filter,
            self.word_number_filter,
        ):
            self._run_operator(operator)

    def _stage_content_integrity(self) -> None:
        for operator in (
            self.colon_end_filter,
            self.sentence_number_filter,
            self.line_end_with_ellipsis_filter,
        ):
            self._run_operator(operator)

    def _stage_text_quality(self) -> None:
        for operator in (
            self.content_null_filter,
            self.mean_word_length_filter,
            self.symbol_word_ratio_filter,
            self.html_entity_filter,
            self.id_card_filter,
            self.no_punc_filter,
            self.special_character_filter,
        ):
            self._run_operator(operator)

    def _stage_noise_detection(self) -> None:
        for operator in (
            self.watermark_filter,
            self.curly_bracket_filter,
            self.capital_words_filter,
            self.lorem_ipsum_filter,
            self.unique_words_filter,
            self.char_number_filter,
        ):
            self._run_operator(operator)

    def _stage_web_feature_filters(self) -> None:
        for operator in (
            self.line_start_with_bulletpoint_filter,
            self.line_with_javascript_filter,
        ):
            self._run_operator(operator)

    def _stage_quality_scoring(self) -> None:
        if self.quality_filter is None:
            self.quality_filter = PairQualFilter(**self.pairqual_kwargs)
        self._run_operator(self.quality_filter)


    def forward(self, stop_stage: int | None = None) -> None:
        stages = (
            (1, self._stage_language_and_initial_cleaning),
            (2, self._stage_deduplication_and_basic_filters),
            (3, self._stage_content_integrity),
            (4, self._stage_text_quality),
            (5, self._stage_noise_detection),
            (6, self._stage_web_feature_filters),
            (7, self._stage_quality_scoring),
        )

        valid_stage_numbers = {stage_number for stage_number, _ in stages}
        if stop_stage is not None and stop_stage not in valid_stage_numbers:
            raise ValueError(
                f"stop_stage must be between 1 and {max(valid_stage_numbers)}, got {stop_stage}."
            )

        for stage_number, runner in stages:
            runner()
            if stop_stage is not None and stage_number == stop_stage:
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the PT text filtering pipeline.")
    parser.add_argument(
        "--stop-stage",
        type=int,
        default=None,
        help=(
            "Stop running the pipeline after the specified stage number. "
            "Use 1 to execute only the initial cleaning and language identification stage."
        ),
    )
    args = parser.parse_args()

    model = PTTextFilter_GPUPipeline()
    model.forward(stop_stage=args.stop_stage)
