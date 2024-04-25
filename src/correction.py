from src.language_models.llm import LLMOutput
from dataclasses import dataclass
from nltk.tokenize import word_tokenize
import editdistance

@dataclass
class Correction:
    original: LLMOutput
    edited: LLMOutput
    comment: LLMOutput

    def edit_distance(self):
        return editdistance.eval(word_tokenize(self.original.text), word_tokenize(self.edited.text))
    
    def is_edited(self):
        return self.original.text != self.edited.text