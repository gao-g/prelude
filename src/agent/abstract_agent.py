from abc import ABC, abstractmethod
from src.language_models.llm import LLM, LLMOutput
from src.correction import Correction
from src.task.abstract_task import Task
from src.utils.color_utils import print_color
from typing import Dict
import numpy as np


class Agent(ABC):
    _llm: LLM
    _task: Task
    can_cheat: bool = False

    def __init__(self, agent_config, task, workspace):
        self._llm = LLM(agent_config.model, name=self.__class__.__name__, workspace=workspace)
        self._task = task
        self.can_cheat = agent_config.agent in {'oracle-preference', 'greedy-rag-v2'}
        self._preference = None
        print_color(f'Agent: {self.__class__.__name__}', color='magenta')

    def cheat(self, environment):   # Only Oracle preference baseline has it overridden
        raise ValueError(f'You are not supposed to cheat with {self.__class__.__name__}')
    
    def metrics(self, message, correction: Correction):
        correction.original = self._llm.get_logprobs(message, correction.original)
        if correction.is_edited():
            correction.edited = self._llm.get_logprobs(message, correction.edited)
        else:
            correction.edited = correction.original
        cost = self._task.cost(correction)
        print_color(f'cost: {cost}', color="red")
        print_color(f'Model response: {correction.original.text}', color="blue")
        print_color(f'User edits: {correction.edited.text}', color="green")
        print_color(f'----------------------------------------------------------- ', color='yellow')
        return {
            'message': message,
            'completion': correction.original.text,
            'completion_logprobs': correction.original.logprobs,
            'completion_token_count': correction.original.token_count,
            'edited': correction.edited.text,
            'edited_logprobs': correction.edited.logprobs,
            'edited_token_count': correction.edited.token_count,
            'cost': cost,
            'comment': correction.comment.text,
            'preference_inference': self._preference
        }
    
    @abstractmethod
    def complete(self, text) -> LLMOutput:
        ...

    @abstractmethod
    def learn(self, message, correction: Correction) -> Dict:
        """
        doing learning based on input message and correction 
        (incapsulated both proposed and edited completion with logprobs/token_counts)
        and producing dictionary of metrics
        """
        ...
    
