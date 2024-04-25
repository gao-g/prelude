from src.agent.abstract_agent import Agent
from src.language_models.llm import LLMOutput
from src.correction import Correction
from typing import Dict

class OraclePreferenceAgent(Agent):
    def __init__(self, agent_config, task, workspace):
        super().__init__(agent_config, task, workspace)
        self._preference = None

    def cheat(self, environment):   # Only Oracle preference baseline has it overridden
        self._preference = environment.user._preference

    def complete(self, text) -> LLMOutput:
        assert self._preference is not None, "oracle_preference.py requires the preference to be set"
        prompt = self._task.get_task_prompt(text, preference=self._preference)
        return self._llm.respond(prompt)

    def learn(self, message, correction: Correction) -> Dict:
        return self.metrics(message, correction)