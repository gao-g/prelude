from datasets import load_dataset
from torch.utils.data import Dataset

from typing import List


def load_data(dataset: str, num_ex: int = -1, split: str = 'train'):
    """
    return an OurInputDataset object with the specified number of examples
    see the choices for split on each dataset below
    """

    if dataset == 'cnn_dailymail':
        # possible splits: train, validation, test
        data = load_dataset(dataset, '3.0.0', split=split)
        # annotation_key='highlights',
        parsed_data = OurInputDataset(data=data, num_ex=num_ex, id_key='id', article_key='article',
                                      doc_type=dataset)
    elif dataset == 'xsum':
        # possible splits: train, validation, test
        data = load_dataset(dataset)[split]
        # annotation_key='summary', 
        parsed_data = OurInputDataset(data=data, num_ex=num_ex, id_key='id', article_key='document',
                                      doc_type=dataset)
    elif dataset == 'slf5k':
        # possible splits: train, development, validation, test
        data = load_dataset("JeremyAlain/SLF5K")[split]
        # annotation_key='ideal_human_summary', 
        parsed_data = OurInputDataset(data=data, num_ex=num_ex, id_key='id', article_key='post',
                                      doc_type=dataset) 
    elif dataset == 'wikipedia':
        # possible splits: train 
        data = load_dataset(dataset, '20220301.simple')[split]
        filtered_data = []
        for ex in data:
            length = len(ex['text'].split()) 
            if length > 500 and length < 700: # 4809
                filtered_data.append(ex)
        parsed_data = OurInputDataset(data=filtered_data, num_ex=num_ex, id_key='id', article_key='text',
                                      doc_type=dataset)
    elif dataset == 'CShorten/ML-ArXiv-Papers':
        # possible splits: train
        data = load_dataset(dataset)[split]
        parsed_data = OurInputDataset(data=data, num_ex=num_ex, id_key='Unnamed: 0.1', article_key='abstract',
                                      doc_type=dataset)
    elif dataset == 'imdb':
        # possible splits: train, test, unsupervised
        data = load_dataset(dataset)[split]
        # note that the label is not actuall ids
        parsed_data = OurInputDataset(data=data, num_ex=num_ex, id_key='label', article_key='text',
                                      doc_type=dataset)
    elif dataset == 'ccby':
        data = load_dataset('orieg/elsevier-oa-cc-by')['train']
        filtered_data = []
        for ex in data:
            if ex['author_highlights'] != []:
                filtered_data.append({'id': len(filtered_data), 
                                      'text': ' My title: ' + ex['title'] + 
                                              '. My abstract: ' + ex['abstract'] +
                                              ' My highlights: ' + '. '.join(ex['author_highlights'])})
        parsed_data = OurInputDataset(data=filtered_data, num_ex=num_ex, id_key='id', article_key='text',
                                      doc_type=dataset) 
    elif dataset == 'ampere':
        data = load_dataset('launch/ampere', split='train')
        data = [{'id': ex['doc_id'], 'text': ' '.join(ex['text'])} for ex in data]
        parsed_data = OurInputDataset(data=data, num_ex=num_ex, id_key='id', article_key='text',
                                      doc_type=dataset) 
    elif dataset == 'paper_tweet':
        data = load_dataset('nitsanb/paper_tweet', split='train')
        # data = load_dataset('nitsanb/paper_tweet', data_files='https://huggingface.co/datasets/nitsanb/paper_tweet/blob/main/paper_tweet_data.csv')
        # data = load_dataset('csv', data_files='https://huggingface.co/datasets/nitsanb/paper_tweet/blob/main/paper_tweet_data.csv')
        filtered_data = []
        for ex in data:
            ex = ex['text']
            if '[' in ex:
                continue
            try:
                # tweet = ex[ex.index('['):(ex.index(']') + 1)]
                tweet = ex[(ex.index('"') + 1):(ex.rfind('"') - 2)]
            except Exception:
                continue 
            filtered_data.append({'id': len(filtered_data), 'text': tweet})
        parsed_data = OurInputDataset(data=filtered_data, num_ex=num_ex, id_key='id', article_key='text',
                                      doc_type=dataset) 
   
    else:
        raise NotImplementedError

    return parsed_data





class OurInputExample:
    """
    Structure for one input example with id, article, and an annotated summary
    """
    def __init__(self, 
                 id: str, 
                 article: str, 
                 #  annotation: str, 
                 doc_type: str,
                 model_summary: str = None, 
                 user_pref: str = None, 
                 user_edits: str = None, 
                 model_refinement: str = None,
                 eval_yesno: str = None, 
                 eval_rationale: str = None):
        """
        Creates one InputExample with the given id, article and annotation
        """
        self.id = id
        self.article = article
        # self.annotation = annotation
        self.doc_type = doc_type
        self.model_summary = model_summary
        self.user_pref = user_pref
        self.user_edits = user_edits
        self.model_refinement = model_refinement
        self.eval_yesno = eval_yesno
        self.eval_rationale = eval_rationale


    def __str__(self):
        return f"<InputExample> article ID {self.id}: {self.article[:100]}..., annotation: {self.annotation}"


class OurInputDataset(Dataset): 
    """
    Structure for a dataset with example ids (article -> annotated summary)
    We create this object to unify different datasets on hugging face which have different formats

    *_keys are string keys in each example dictionary in the given data loaded from hugging face
    """
    def __init__(self, data, num_ex: int, id_key: str, article_key: str, doc_type: str = ''):
        from itertools import islice
        if num_ex > 0:
            data = islice(data, num_ex)

        self.dataset = [OurInputExample(id=d[id_key],
                                article=d[article_key],
                                doc_type=doc_type) for d in data]

    def __getitem__(self, item):        
        return self.dataset[item]
    
    def __len__(self):
        return len(self.dataset)
