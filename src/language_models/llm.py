import logging
from dataclasses import dataclass, field
from typing import Sequence, Optional, Union

from src.language_models.azure_gpt import AzureGPT
from src.workspace.workspace import Workspace


@dataclass
class LLMOutput:
    text: str
    logprobs: Optional[Sequence[float]] = None

    @property
    def token_count(self):
        return len(self.logprobs) if self.logprobs is not None else None


class ChatBasedPromptWrapper:
    def __init__(
        self, user_prompt: str, system_prompt: str = "You are an AI assistant that helps people find information."
    ):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def to_list(self) -> list[dict[str, str]]:
        chat_log = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]
        return chat_log


class LLM:
    def __init__(self, model, name="llm", workspace=None, chat_prompt_wrapper=ChatBasedPromptWrapper) -> None:
        self.impl = AzureGPT(model)
        self.name = name
        self.workspace = workspace
        self.chat_prompt_wrapper = chat_prompt_wrapper

    def respond(
        self,
        input_prompt: Union[str, list],
        log=None,
        max_tokens=4000,
        temperature=0.0,
        max_attempts=10000,
        expected_finish_reason="stop",
    ) -> LLMOutput:
        if self.impl.name != "gpt-35-turbo-instruct":
            if type(input_prompt) == str:
                input_prompt = self.chat_prompt_wrapper(input_prompt).to_list()
            assert type(input_prompt) == list
            response = self.impl.get_response_given_chat_completion_prompt(
                input_prompt,
                temperature=temperature,
                max_attempt=max_attempts,
                max_tokens=max_tokens,
                expected_finish_reason=expected_finish_reason,
            )
        else:
            assert type(input_prompt) == str
            response = self.impl.get_response_given_completion_prompt(
                input_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                max_attempt=max_attempts,
                expected_finish_reason=expected_finish_reason,
            )
        if self.workspace:
            self.workspace.log_message(input_prompt, self.name, response)
        return LLMOutput(response)

    def get_logprobs(self, text, output, temperature=0.0, max_attempts=10000) -> LLMOutput:
        """
        Returns LLMOutput with given output as text and updated logprobs/token_count
        """
        output = output.text if isinstance(output, LLMOutput) else output
        return LLMOutput(text=output, logprobs=[])

    @staticmethod
    def check_last_text_token(text: str, text_token_count: int, token: str, logprobs) -> bool:
        txt_last_token = logprobs.tokens[text_token_count - 1].strip()
        start, end = text.rfind(txt_last_token), len(text)
        return start != -1 and text[start:end].strip() == txt_last_token
