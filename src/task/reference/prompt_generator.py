from pathlib import Path


class PromptGenerator:
    def __init__(self):
        self.type2pref = {'cnn_dailymail': 'academic writing style, formal English, avoiding contractions',
                          'slf5k': 'journalistic, neutral tone, engaging, story-telling, narrative, balanced use of adjectives',
                          'wikipedia': 'no jargon, accessible to kids, include relatable examples, engaging, capitalization for emphasis',
                          'CShorten/ML-ArXiv-Papers': 'tweet style, simple English, casual tone, inquisitive, skillful foreshadowing',
                          'imdb': 'bullet points, brief, succinct, active voice, not emotive, not starting with capital letters',
                          }
        
    
    def generate_prompt_to_get_summary(self, article):
        prompt_template_get_summary = """ Article:{ARTICLE} \n
        Please summarize the above article: """
        prompt = prompt_template_get_summary.replace('{ARTICLE}', article)
        return prompt

    def generate_prompt_to_get_summary_icl(self, ex_list, article):
        prompt = ''
        for ex in ex_list:
            prompt = prompt + 'Original summary of an article: ' + ex['original'] + '\n'
            prompt = prompt + 'Revised summary by a user: ' + ex['revised'] + '\n\n'
        prompt += """Article: {ARTICLE} \n
        Based on the edits and revision by this user on the original summary in the above examples, 
        Please summarize the above article: """
        return prompt.replace('{ARTICLE}', article)
    
    def generate_prompt_to_get_summary_based_on_user_pref(self, article, user_pref):
        prompt_template_get_summary_based_on_user_pref = """Article:{ARTICLE} \n
        Assume that you need to summarize the above article for a user, 
        who prefers the following style: {USER_PREF}. 
        Please write a summary of the above article to address those specified preferences."""
        prompt = prompt_template_get_summary_based_on_user_pref.replace('{ARTICLE}', article).replace('{USER_PREF}', user_pref)
        return prompt
    
    def generate_prompt_to_infer_user_pref(self, ex_list):
        prompt = ''
        for ex in ex_list:
            prompt = prompt + 'Original summary of an article: ' + ex['original'] + '\n'
            prompt = prompt + 'Revised summary by a user: ' + ex['revised'] + '\n\n'
        prompt += """Based on the edits and revision by this user on the original summary in the above examples, 
        what do you find about this user's generic preference in terms of writing style and formatting?  
        Please answer by listing keywords.
        """
        return prompt 

    def generate_prompt_from_doc_to_get_edits(self, input_article, model_output, doc_type):
        user_pref = self.type2pref[doc_type]
        prompt_template_get_edits_for_convergence = """ARTICLE: {ARTICLE} \n
        Summary: {SUMMARY}\n
        Is the above summary of the above article good for a user who prefers {PREFERENCE}? Please answer yes or no."""
        prompt1 = prompt_template_get_edits_for_convergence.replace('{ARTICLE}', input_article).replace('{SUMMARY}', model_output).replace('{PREFERENCE}', user_pref)
        prompt_template_get_edits = """Summary: {SUMMARY} \n
        Assume that you prefer {PREFERENCE}. 
        Please revise the above summary of an article to meet your style:"""
        prompt2 = prompt_template_get_edits.replace('{ARTICLE}', input_article).replace('{SUMMARY}', model_output).replace('{PREFERENCE}', user_pref)
        return prompt1, prompt2


class OldPromptGenerator:
    def __init__(self):

        self.type2pref = {'cnn_dailymail': 'academic writing style, formal English, avoiding contractions.',
                          'slf5k': 'journalistic, neutral tone, engaging, story-telling, narrative, balanced use of adjectives.',
                          'wikipedia': 'no jargon, accessible to kids, include relatable examples, engaging, capitalization for emphasis.',
                          'CShorten/ML-ArXiv-Papers': 'tweet style, simple English, casual tone, inquisitive, skillful foreshadowing.',
                          'imdb': 'bullet points, brief, succinct, active voice, not emotive, not starting with capital letters.',
                          }

        filename = 'files/user_preferences.txt'
        with open(filename, 'r') as f:
            lines = [line.rstrip() for line in f]
        self.user_prefs = lines

        # a list of strings: each is a user preference
        filename = 'files/inferred_user_preferences_keywords_filtered.txt'
        with open(filename, 'r') as f:
            lines = [line.rstrip() for line in f]
        self.defined_user_prefs = lines

        prompt_template_filename = 'files/prompt_get_summary.txt'
        self.prompt_template_get_summary = Path(prompt_template_filename).read_text()
        prompt_template_filename = 'files/prompt_get_edits.txt'
        self.prompt_template_get_edits = Path(prompt_template_filename).read_text()
        # prompt_template_filename = 'files/prompt_baseline_in-context-learning_with_edits_example_pref0.txt'
        # self.prompt_template_get_summary_icl = [Path(prompt_template_filename).read_text()]
        # self.prompt_template_get_summary_icl = [''] * 18
        # for idx in [0, 4, 6, 10, 13]:
        #     prompt_template_filename = f'files/prompt_baseline_in-context-learning_with_edits_example_pref{idx}.txt'
        #     self.prompt_template_get_summary_icl[idx] = Path(prompt_template_filename).read_text()

        prompt_template_filename = 'files/prompt_get_edits_for_convergence.txt'
        self.prompt_template_get_edits_for_convergence = Path(prompt_template_filename).read_text()

        prompt_template_filename = 'files/prompt_get_edits_one_step.txt'
        self.prompt_template_get_edits_one_step = Path(prompt_template_filename).read_text()

        prompt_template_filename = 'files/prompt_infer_preference.txt'
        self.prompt_infer_preference = Path(prompt_template_filename).read_text()
        prompt_template_filename = 'files/prompt_get_summary_based_on_user_pref.txt'
        self.prompt_template_get_summary_based_on_user_pref = Path(prompt_template_filename).read_text()

        prompt_template_filename = 'files/_prompt_get_summary_with_gold_pref.txt'
        self.prompt_template_get_summary_with_gold_pref = Path(prompt_template_filename).read_text()

        prompt_template_filename = 'files/_prompt_get_summary_with_defined_pref.txt'
        self.prompt_template_get_summary_with_defined_pref = Path(prompt_template_filename).read_text()

        prompt_template_filename = 'files/prompt_aggregate_preference.txt'
        self.prompt_aggregate_preference = Path(prompt_template_filename).read_text()

    def generate_prompt_to_aggregate_user_pref(self, prefs):
        prompt = ''
        for pref in prefs:
            prompt = prompt + pref + '\n'
        return prompt + self.prompt_aggregate_preference

    # used in new code 
    def generate_prompt_to_infer_user_pref(self, ex_list):
        prompt = ''
        for ex in ex_list:
            prompt = prompt + 'Original summary of an article: ' + ex['original'] + '\n'
            prompt = prompt + 'Revised summary by a user: ' + ex['revised'] + '\n\n'

        return prompt + self.prompt_infer_preference
        # prompt = 'Writing samples from a user:'
        # idx = 1
        # for ex in ex_list:
        #     prompt = prompt + f'Sample {idx}:' + ex['revised'] + '\n'
        #     idx += 1
        # prompt = prompt + "Based on the above writing samples, what do you find about this user's generic preference in terms of writing style and formatting?  Please answer by listing keywords."
        # return prompt 
    
    # used in new code 
    def generate_prompt_to_get_summary_based_on_user_pref(self, article, user_pref):
        prompt = self.prompt_template_get_summary_based_on_user_pref.replace('{ARTICLE}', article).replace('{USER_PREF}', user_pref)
        return prompt

    # used in new code
    def generate_prompt_to_get_summary(self, ex):
        prompt = self.prompt_template_get_summary.replace('{ARTICLE}', ex.article)
        return prompt

    def generate_prompt_to_get_summary_with_gold_pref(self, ex, pref_idx):
        prompt = self.prompt_template_get_summary_with_gold_pref.replace('{ARTICLE}', ex.article).replace('{USER_PREF}', self.user_prefs[pref_idx])
        return prompt 

    def generate_prompt_to_get_summary_with_defined_pref(self, ex, pref_idx):
        # prompt = self.prompt_template_get_summary_with_defined_pref.replace('{ARTICLE}', ex.article).replace('{USER_PREF}', self.defined_user_prefs[pref_idx])
        
        # prompt = self.prompt_template_get_summary_with_defined_pref.replace('{ARTICLE}', ex.article).replace('{USER_PREF}', 'bullet points, tweet, academic, kid-friendly, shorter sentences')
        prompt = self.prompt_template_get_summary_with_defined_pref.replace('{ARTICLE}', ex.article).replace('{USER_PREF}', 'introducing this article to elementary school students,  in simple language that is easy to understand and engaging')
        return prompt 
    
    def generate_prompt_to_get_summary_icl(self, ex, pref_idx):
        prompt = self.prompt_template_get_summary_icl[pref_idx].replace('{ARTICLE}', ex.article)
        return prompt

    def generate_prompt_to_get_edits_one_step(self, model_output, pref_idx):
        user_pref = self.user_prefs[pref_idx]
        prompt = self.prompt_template_get_edits_one_step.replace('{SUMMARY}', model_output).replace('{PREFERENCE}', user_pref)
        return prompt
    
    def generate_prompt_to_get_edits(self, model_output, pref_idx):
        user_pref = self.user_prefs[pref_idx]
        prompt = self.prompt_template_get_edits.replace('{SUMMARY}', model_output).replace('{PREFERENCE}', user_pref)
        return prompt

    def generate_prompt_from_pref_to_get_edits_for_convergence(self, input_article, model_output, pref_idx):
        user_pref = self.user_prefs[pref_idx]
        prompt1 = self.prompt_template_get_edits_for_convergence.replace('{ARTICLE}', input_article).replace('{SUMMARY}', model_output).replace('{PREFERENCE}', user_pref)
        prompt2 = self.prompt_template_get_edits.replace('{ARTICLE}', input_article).replace('{SUMMARY}', model_output).replace('{PREFERENCE}', user_pref)
        return prompt1, prompt2

    # used in new code
    def generate_prompt_from_doc_to_get_edits(self, input_article, model_output, doc_type):
        user_pref = self.type2pref[doc_type]
        prompt1 = self.prompt_template_get_edits_for_convergence.replace('{ARTICLE}', input_article).replace('{SUMMARY}', model_output).replace('{PREFERENCE}', user_pref)
        prompt2 = self.prompt_template_get_edits.replace('{ARTICLE}', input_article).replace('{SUMMARY}', model_output).replace('{PREFERENCE}', user_pref)
        return prompt1, prompt2
