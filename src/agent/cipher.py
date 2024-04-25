from src.agent.abstract_agent import Agent
from src.agent.encoders.encoder_wrapper import EncoderWrapper
from src.language_models.llm import LLMOutput
from src.agent.rag import RAG
from src.correction import Correction
from typing import Dict


class Cipher1Agent(Agent):
    def __init__(self, agent_config, task, workspace, encoder_type='bert'):
        # Create an encoder model to encode documents
        encoder_model = EncoderWrapper().make_encoder(encoder_type)
        encode = lambda doc: encoder_model.encode(doc).view(-1)
        self.rag = RAG(encode)
        # Current doc
        self.rag_doc = None

        super().__init__(agent_config, task, workspace)


    def complete(self, text) -> LLMOutput:
        if len(self.rag) > 0:
            closest_replay_doc_encoding, self.rag_doc, self._preference  = self.rag.get(text)[0]

        prompt = self._task.get_task_prompt(text, preference=self._preference)
        return self._llm.respond(prompt)

    def learn(self, message, correction: Correction) -> Dict:
        if correction.edited != correction.original:
            prompt = self._task.get_preference_inference_prompt([correction])
            self._preference = self._llm.respond(prompt).text

        self.rag.add(message, self._preference)

        return dict(self.metrics(message, correction), rag_doc=self.rag_doc)


class CipherNAgent(Agent):
    def __init__(self, agent_config, task, workspace, encoder_type='bert', icl_count=3):
        # Create an encoder model to encode documents
        encoder_model = EncoderWrapper().make_encoder(encoder_type)
        encode = lambda doc: encoder_model.encode(doc).view(-1)
        self.rag = RAG(encode)
        self.icl_count = icl_count
        # Current doc
        self.rag_docs = None
        self._preferences = None
        self._pref_aggregated = None
        super().__init__(agent_config, task, workspace)


    def complete(self, text) -> LLMOutput:
        if len(self.rag) > 0:
            _dp = self.rag.get(text, topk=self.icl_count)
            self.rag_docs, self._preferences  = [doc for _, doc, __ in _dp], [pref for _, __, pref in _dp]

        if self._preferences:
            prompt = self._task.get_majority_preference_prompt(self._preferences)
            self._pref_aggregated = self._llm.respond(prompt).text

        prompt = self._task.get_task_prompt(text, preference=self._pref_aggregated)
        return self._llm.respond(prompt)

    def learn(self, message, correction: Correction) -> Dict:
        self._preference = self._pref_aggregated
        if correction.edited != correction.original:
            prompt = self._task.get_preference_inference_prompt([correction])
            self._preference = self._llm.respond(prompt).text

        self.rag.add(message, self._preference)

        return dict(self.metrics(message, correction), rag_preferences=self._preferences, rag_docs = self.rag_docs, preference_aggregated = self._pref_aggregated)

