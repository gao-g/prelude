import os
import time
import openai
from azure.identity import AzureCliCredential, get_bearer_token_provider
import logging
from typing import Union, Dict, List


class AzureGPT:
#    MAX_SIZE = 16300

    def __init__(self, name: str = "gpt-35-turbo-instruct", credential = None):
        token_provider = get_bearer_token_provider(credential or AzureCliCredential(), "https://cognitiveservices.azure.com/.default")
        self.name = name
        self.dummy = False
        if name == "gpt-35-turbo-instruct" or name == "gpt-35-turbo" or name == "gpt-4":
            self.client = openai.AzureOpenAI(
                api_version="2023-05-15",
                azure_endpoint=os.getenv("GCR_GPT_URL"),
                azure_ad_token_provider=token_provider
            )

        elif name == "dummy":
            self.dummy = True

        else:
            raise NotImplementedError

    def get_response_given_completion_prompt(
        self, prompt: str, temperature: float = 0.0, max_attempt=10000, max_tokens=300, expected_finish_reason="stop"
    ):
        if self.dummy:
            return prompt[:100]

    #    if len(prompt) > AzureGPT.MAX_SIZE:
    #        logging.warning(
    #            f"get_response_given_completion_prompt: truncating prompt (len:{len(prompt)})- consider increasing AzureGPT.MAX_SIZE"
    #        )
    #        prompt = prompt[-AzureGPT.MAX_SIZE :]

        if self.name == "gpt-35-turbo-instruct":
            oai_args = {
                "model": self.name,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            response = AzureGPT.retry_openai_func(
                self.client.completions.create,
                expected_finish_reason=expected_finish_reason,
                max_attempt=max_attempt,
                **oai_args,
            )
            return response.choices[0].text.strip()
        else:
            raise NotImplementedError

    def get_response_given_chat_completion_prompt(
        self,
        chat_log: List[Dict],
        temperature: float = 0.0,
        max_attempt=10000,
        max_tokens=300,
        expected_finish_reason="stop",
    ):
        if self.dummy:
            return chat_log

        sum_len = sum(len(entry["content"]) for entry in chat_log if "content" in entry)
    #    if sum_len > AzureGPT.MAX_SIZE:
    #        logging.warning(
    #            f"get_response_given_chat_completion_prompt: truncating prompt (len:{sum_len})- consider increasing AzureGPT.MAX_SIZE"
    #        )
            # todo: truncate chat log somehow

        if self.name == "gpt-4" or self.name == "gpt-35-turbo":
            oai_args = {
                "model": self.name,
                "messages": chat_log,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": 0.95,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "stop": None,
            }
            response = AzureGPT.retry_openai_func(
                self.client.chat.completions.create,
                expected_finish_reason=expected_finish_reason,
                max_attempt=max_attempt,
                **oai_args,
            )
            return response.choices[0].message.content.strip()

        else:
            raise NotImplementedError


    def get_logprobs(self, prompt, temperature: float = 0.0, max_attempt=10000):
        if self.name != "gpt-35-turbo-instruct":
            raise NotImplementedError

        oai_args = {
            "model": self.name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": 0,
            "logprobs": 1,
            "echo": True,
        }
        response = AzureGPT.retry_openai_func(
            self.client.completions.create,
            "length",
            max_attempt=max_attempt,
            **oai_args,
        )

        return response.choices[0].logprobs

    def get_prompt_length(self, prompt, temperature: float = 0.0, max_attempt=10000):
        logprobs = self.get_logprobs(prompt, temperature=temperature, max_attempt=max_attempt)
        return len(logprobs.token_logprobs)

    @staticmethod
    def retry_openai_func(func, expected_finish_reason="stop", max_attempt=10000, **kwargs):
        import regex as re
        attempts = 0
        sleep_time = 10
        while True:
            try:
                response = func(**kwargs)
                if response.choices[0].finish_reason not in ("stop", "length"):
                    logging.warning(
                        f"retry_openai_func: successful openai response but finish_reason is not 'stop' or 'length'. actual= {response.choices[0].finish_reason}. retrying.."
                    )
                    time.sleep(10)
                else:
                    if response.choices[0].finish_reason == "length" and expected_finish_reason == "stop":
                        logging.warning("retry_openai_func: reached max tokens - consider increasing max_tokens")
                    break
            except Exception as e:
                logging.error(
                    f"retry_openai_func: call {func.__module__}.{func.__name__}: attempt {attempts} failed {e}"
                )
                r = re.search(r'Please retry after\s(\d+)', str(e))
                sleep_time = int(r.group(1)) if r else 10
                attempts += 1
                if attempts == max_attempt:
                    logging.critical(f"retry_openai_func: reached max attempt ({max_attempt}). failing")
                    return ""
                else:
                    time.sleep(sleep_time)
        #    except Exception as e:
        #        logging.critical(
        #            f"retry_openai_func: call {func.__module__}.{func.__name__}: attempt {attempts} failed {e}\nMessages:\n{kwargs.get('messages', '')}"
        #        )
        #        raise
        return response
