from src.language_models.llm import LLMOutput
from src.correction import Correction

class Environment:
    def __init__(self, user):
        self.user = user

    def __iter__(self):
        return self.user.next_message()

    def user_edit(self, completion: LLMOutput) -> Correction:
        return self.user.edit(completion)
    
    def debug_metrics(self):
        return self.user.debug_metrics()