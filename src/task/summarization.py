from src.task.abstract_task import Task
from src.task.dataset_helpers import load_data
from src.task.cost import get_cost_func
from src.correction import Correction

import numpy as np
from typing import Tuple, Iterable, Optional, List

class Summarization(Task):
    type_2_pref = {
        'cnn_dailymail': 'style targeted to young children, storytelling, short sentences, playful language, interactive, positive',
        'slf5k': 'second person narrative, brief, show emotions, invoke personal reflection, immersive',
        'wikipedia': 'bullet points, parallel structure, brief',
        'CShorten/ML-ArXiv-Papers': 'tweet style, simple English, inquisitive, skillful foreshadowing, with emojis',
        'imdb': 'question answering style',
    }

    def __init__(self, task_config):
        self._data = Summarization._get_dataset(
            task_config.datasets or ['cnn_dailymail', 'slf5k', 'wikipedia', 'CShorten/ML-ArXiv-Papers', 'imdb'],
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
            d.user_pref = Summarization.type_2_pref[d.doc_type]
        return result

    def next(self) -> Iterable[Tuple[str, str]]:
        """
        Iterating over tuples (mext_message, user_preferenceW)
        """
        for d in self._data:
            yield d.article, d.user_pref


    def get_edit_prompts(self, input: str, output: str, preference: str) -> Tuple[str, str]:
        resolution_prompt = "\n".join([
            f"Article:\n{input}",
            f"Summary:\n{output}",
            f"Is the above summary of the above article good for person who would love to use the following style: {preference}? Please answer yes or no."])
        edit_prompt = "\n".join([
            f"Summary:\n{output}",
            f"Assume that you prefer the following style: {preference}.",
            f"Please revise the above summary of an article to meet your style:"])
        return resolution_prompt, edit_prompt
    
    def get_task_prompt(self, input: str, preference: Optional[str] = None) -> str:
        if preference is None:
            return "\n".join([
                f"Article:\n{input}",
                f"Please summarize the above article."])
        return "\n".join([
            f"Article:\n{input}",
            f"Assume that you need to summarize the above article for a user, who prefers the following style: {preference}.",
            f"Please write a summary of the above article to address those specified preferences."])

    def get_task_prompt_icl(self, input: str, corrections: List[Correction]) -> str:
        prompt = ''
        for correction in corrections:
            prompt = prompt + f'Original summary of an article:\n{correction.original.text}\n'
            prompt = prompt + f'Revised summary by a user:\n{correction.edited.text}\n\n'
        prompt += "\n".join([
            f"Article:\n{input}",
            f"Based on the edits and revision by this user on the original summary in the above examples, please summarize the above article."])
        return prompt
    
    def get_task_prompt_icl_pref(self, input: str, preferences: List[str]) -> str:
        prompt = 'List of user preferences successfully being used to generate summaries of similar documents:\n'
        for preference in preferences:
            prompt = prompt + f'- {preference}\n'
        prompt += "\n".join([
            f"Article:\n{input}",
            f"Using the qualities most represented in the above list of preferences, please summarize the above article."])
        return prompt

    def get_preference_inference_prompt(self, corrections: List[Correction]) -> str:
        prompt = ''
        for correction in corrections:
            prompt = prompt + f'Original summary of an article: {correction.original.text}\n'
            prompt = prompt + f'Revised summary by a user: {correction.edited.text}\n\n'
        prompt += "\n".join([
            f"Based on the edits and revision by this user on the original summary in the above examples, what do you find about this user's generic preference in terms of writing style and formatting?",  
            f"Please answer in a short phrase and only recommend those preferences that are widely used."])
        return prompt 


    def get_majority_preference_prompt(self, preferences: List[str]) -> str:
        prompt = 'List of user preferences successfully being used to generate summaries of similar documents: \n'
        for preference in preferences:
            prompt += f'- {preference}\n'
        prompt += "Based on the the above examples, please come up with short phrase with the most represented summarization preferences of the user."
        return prompt 