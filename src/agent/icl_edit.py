from src.agent.abstract_agent import Agent
from src.language_models.llm import LLMOutput
from src.correction import Correction
from src.agent.encoders.encoder_wrapper import EncoderWrapper
from src.agent.rag import RAG
from typing import Dict
import numpy as np

class IclEditAgent(Agent):
    def __init__(self, agent_config, task, workspace, encoder_type='bert', icl_count=3):
        encoder_model = EncoderWrapper().make_encoder(encoder_type)
        encode = lambda doc: encoder_model.encode(doc).view(-1)
        self.rag = RAG(encode)
        self.icl_count = icl_count
        super().__init__(agent_config, task, workspace) 

    def complete(self, text) -> LLMOutput:
        preference = None
        examples = [correction for _, __, correction in self.rag.get(text, topk=self.icl_count)]
        if any(examples):
            prompt = self._task.get_task_prompt_icl(text, examples)
            return self._llm.respond(prompt)
        prompt = self._task.get_task_prompt(text, preference)
        return self._llm.respond(prompt)

    def learn(self, message, correction: Correction) -> Dict:
        self.rag.add(message, correction)
        return self.metrics(message, correction)
    