from src.task.abstract_task import Task
from src.task.dataset_helpers import load_data
from src.task.cost import get_cost_func
from src.correction import Correction

import numpy as np
from typing import Tuple, Iterable, Optional, List

class EmailWriting(Task):
    type_2_pref = {
        'ccby': 'structured, straight to the points, respectful, professional greeting and closing',
        'slf5k': 'informal, conversational, no closing',
        'ampere': 'casual tone, positive, clear, call to action',
        'paper_tweet': 'engaging, personalized, professional tone, thankful closing',
    }

    def __init__(self, task_config):
        self._data = EmailWriting._get_dataset(
            task_config.datasets or ['slf5k', 'ccby', 'ampere', 'paper_tweet'],
            task_config.num_train_ex, task_config.seed, )
        self._cost = get_cost_func(task_config.cost) 

    @staticmethod
    def _get_dataset(datasets, num_train_ex, seed):
        from itertools import chain, islice
        result = []
        num_doc_types = len(datasets)
        num_ex_per_doc_type = int(num_train_ex / num_doc_types)
        for dataset in datasets:
            result.append(list(load_data(dataset=dataset,
                                    num_ex=-1,
                                    split='train')))
        rng = np.random.default_rng(seed=seed)
        for r in result:
            rng.shuffle(r)
        result = list(chain.from_iterable(map(lambda r: islice(r, num_ex_per_doc_type), result)))
        rng.shuffle(result)
        for d in result:
            d.user_pref = EmailWriting.type_2_pref[d.doc_type]
        return result

    def next(self) -> Iterable[Tuple[str, str]]:
        """
        Iterating over tuples (mext_message, user_preferenceW)
        """
        for d in self._data:
            yield d.article, d.user_pref

    def get_edit_prompts(self, input: str, output: str, preference: str) -> Tuple[str, str]:
        resolution_prompt = "\n".join([
            f"Notes:\n{input}",
            f"Email:\n{output}",
            f"Is the above email based on the above notes good for a user who wants the following style: {preference}? Please answer yes or no."])
        edit_prompt = "\n".join([
            f"Email:\n{output}",
            f"Assume that you prefer the following style: {preference}.",
            f"Please revise the above email to meet your style."])
        return resolution_prompt, edit_prompt
    
    def get_task_prompt(self, input: str, preference: Optional[str] = None) -> str:
        if preference is None:
            return "\n".join([
                f"Notes:\n{input}",
                f"Please write a short email based on your above notes."])
        return "\n".join([
            f"Notes:\n{input}",
            f"These notes are written by a user who prefers the following style of emails: {preference}.", 
            f"Please write a short email based on the above notes to address those specified preferences."])

    def get_task_prompt_icl(self, input: str, corrections: List[Correction]) -> str:
        prompt = ''
        for correction in corrections:
            prompt = prompt + f'Original email:\n{correction.original.text}\n'
            prompt = prompt + f'Revised email:\n{correction.edited.text}\n\n'
        prompt += "\n".join([
            f"Notes:\n{input}",
            f"Based on the edits and revision by this user on the original email in the above examples, please write an email based on the above notes for this user."])
        return prompt
    
    def get_task_prompt_icl_pref(self, input: str, preferences: List[str]) -> str:
        prompt = 'List of user preferences successfully being used to generate emails of a similar kind:\n'
        for preference in preferences:
            prompt = prompt + f'- {preference}\n'
        prompt += "\n".join([
            f"Notes:\n{input}",
            f"Using the qualities most represented in the above list of preferences, please write an email based on the above notes."])
        return prompt

    def get_preference_inference_prompt(self, corrections: List[Correction]) -> str:
        prompt = ''
        for correction in corrections:
            prompt = prompt + f'Original email:\n{correction.original.text}\n'
            prompt = prompt + f'Revised email:\n{correction.edited.text}\n\n'
        prompt += "\n".join([
            f"Based on the edits and revision by this user on the original email in the above examples, what do you find about this user's generic preference in terms of writing style and formatting?",
            f"Please answer in a short phrase and only recommend those preferences that are widely used."])
        return prompt 


    def get_majority_preference_prompt(self, preferences: List[str]) -> str:
        prompt = 'List of user preferences successfully being used to generate emails of a similar kind:\n'
        for preference in preferences:
            prompt += f'- {preference}\n'
        prompt += "Based on the the above examples, please come up with short phrase with the most represented writing preferences of this user."
        return prompt 
