from abc import ABC, abstractmethod
from typing import Tuple
from src.correction import Correction


class Task(ABC):
    @abstractmethod
    def next(self) -> Tuple[str, str]:
        """
        Iterating over tuples (mext_message, user_preferenceW)
        """
        ...
    
    @abstractmethod
    def get_edit_prompts(self, input, output, preference) -> Tuple[str, str]:
        """
        Get tuple of prompts:
        - To get yes/no acceptance message
        - To get comments/revisions (if no previously)
        """
        ...
    @abstractmethod
    def get_task_prompt(self, input, preference=None) -> str:
        """
        Get primary prompt for the task
        """

    @abstractmethod
    def get_task_prompt_icl(self, input, corrections):
        """
        Get in-context-learning prompt for the task based on the history of corrections
        """
        ...

    @abstractmethod
    def get_preference_inference_prompt(self, corrections):
        """
        Get preference inference prompt for the task based on the history of corrections
        """
        ...

    def cost(self, correction: Correction):
        """
        Compute cost of the correction
        """
        return self._cost(correction)