from src.agent.abstract_agent import Agent
from src.language_models.llm import LLMOutput
from src.correction import Correction
from typing import Dict


class ContinualLPIAgent(Agent):
    def __init__(self, agent_config, task, workspace, icl_count=3):
        self.icl_count = icl_count
        self.history = []
        super().__init__(agent_config, task, workspace) 

    def complete(self, text) -> LLMOutput:
        if any(self.history):
            prompt = self._task.get_preference_inference_prompt(self.history[-self.icl_count:])
            self._preference = self._llm.respond(prompt).text
        prompt = self._task.get_task_prompt(text, self._preference)
        return self._llm.respond(prompt)

    def learn(self, message, correction: Correction) -> Dict:
        self.history.append(correction)
        return self.metrics(message, correction)