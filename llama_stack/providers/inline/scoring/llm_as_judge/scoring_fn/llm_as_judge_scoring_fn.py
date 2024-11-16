# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from llama_stack.apis.inference.inference import Inference

from llama_stack.providers.utils.scoring.base_scoring_fn import BaseScoringFn
from llama_stack.apis.scoring_functions import *  # noqa: F401, F403
from llama_stack.apis.scoring import *  # noqa: F401, F403
from llama_stack.apis.common.type_system import *  # noqa: F403
import re

from .fn_defs.llm_as_judge_base import llm_as_judge_base


class LlmAsJudgeScoringFn(BaseScoringFn):
    """
    A scoring_fn that assigns
    """

    def __init__(self, inference_api: Inference, *arg, **kwargs) -> None:
        super().__init__(*arg, **kwargs)
        self.inference_api = inference_api
        self.supported_fn_defs_registry = {
            llm_as_judge_base.identifier: llm_as_judge_base,
        }

    async def score_row(
        self,
        input_row: Dict[str, Any],
        scoring_fn_identifier: Optional[str] = None,
        scoring_params: Optional[ScoringFnParams] = None,
    ) -> ScoringResultRow:
        assert (
            scoring_fn_identifier is not None
        ), "Scoring function identifier not found."
        fn_def = self.supported_fn_defs_registry[scoring_fn_identifier]

        # override params if scoring_params is provided
        if scoring_params is not None:
            fn_def.params = scoring_params

        assert fn_def.params is not None, f"LLMAsJudgeparams not found for {fn_def}."
        assert (
            fn_def.params.prompt_template is not None
        ), "LLM Judge prompt_template not found."
        assert (
            fn_def.params.judge_score_regexes is not None
        ), "LLM Judge judge_score_regexes not found."

        input_query = input_row["input_query"]
        expected_answer = input_row["expected_answer"]
        generated_answer = input_row["generated_answer"]

        judge_input_msg = fn_def.params.prompt_template.format(
            input_query=input_query,
            expected_answer=expected_answer,
            generated_answer=generated_answer,
        )

        judge_response = await self.inference_api.chat_completion(
            model_id=fn_def.params.judge_model,
            messages=[
                {
                    "role": "user",
                    "content": judge_input_msg,
                }
            ],
        )
        content = judge_response.completion_message.content
        rating_regexes = fn_def.params.judge_score_regexes

        judge_rating = None
        for regex in rating_regexes:
            match = re.search(regex, content)
            if match:
                judge_rating = match.group(1)
                break

        return {
            "score": judge_rating,
            "judge_feedback": content,
        }

    async def aggregate(
        self, scoring_results: List[ScoringResultRow]
    ) -> Dict[str, Any]:
        # TODO: this needs to be config based aggregation, and only useful w/ Jobs API
        return {}