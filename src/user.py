from src.language_models.llm import LLM, LLMOutput
from src.correction import Correction
from src.task.abstract_task import Task
from src.task.summarization import Summarization
from src.task.email_writing import EmailWriting

def create_task(task_config):
    if task_config.task == 'summarization':
        return Summarization(task_config)
    elif task_config.task == 'email':
        return EmailWriting(task_config)
    else:
        raise ValueError('Unknown task: {task_config.task}')


class User:
    _llm: LLM
    _preference: str
    _message: str
    task: Task

    def __init__(self, user_config, task_config, workspace):
        self._llm = LLM(user_config.model, 'user', workspace)
        self.task = create_task(task_config)

    def next_message(self):
        for message, preference in self.task.next():
            message = message.strip()
            self._message = message
            self._preference = preference
            yield message

    def edit(self, completion: LLMOutput) -> Correction:
        resolution_prompt, edit_prompt = self.task.get_edit_prompts(self._message, completion.text, self._preference)
        resolution = self._llm.respond(resolution_prompt)
        edited = completion
        if 'yes' not in resolution.text.lower():
            edited = self._llm.respond(edit_prompt)
            if 'no' not in resolution.text.lower():
                print('user decision not yes or no:', resolution.text)
        return Correction(original=completion, edited=edited, comment=resolution)
    
    def debug_metrics(self):
        return {'preference_groundtruth': self._preference}
