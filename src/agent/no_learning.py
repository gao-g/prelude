from src.agent.abstract_agent import Agent
from src.language_models.llm import LLMOutput
from src.correction import Correction
from typing import Dict

class NoLearningAgent(Agent):
    def __init__(self, agent_config, task, workspace):
        super().__init__(agent_config, task, workspace)

    def complete(self, text) -> LLMOutput:
        prompt = self._task.get_task_prompt(text)
        return self._llm.respond(prompt)

    def learn(self, message, correction: Correction) -> Dict:
        return self.metrics(message, correction)