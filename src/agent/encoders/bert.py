import torch

from transformers import AutoTokenizer
from transformers import BertModel
from src.agent.encoders.abstract_encoder import AbstractEncoder


class BertEncoding(AbstractEncoder):

    def __init__(self):

        super(AbstractEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")

        if torch.cuda.is_available():
            self.device = "cuda"
            self.model.cuda()
        else:
            self.device = "cpu"

    def encode(self, text):

        batch_token_ids = self.tokenizer(text,
                                         padding="longest",
                                         return_tensors="pt",
                                         truncation=True,
                                         max_length=512).to(self.device)

        results = self.model(**batch_token_ids)

        hidden_state = results[0]       # batch x max_len x hidden_dim
        attention_mask = batch_token_ids.attention_mask.unsqueeze(2)     # batch x max_len x 1
        num_tokens = attention_mask.sum(dim=1)                           # batch x 1

        masked_hidden_state = torch.sum(hidden_state * attention_mask, dim=1)    # batch x hidden_dim
        avg_masked_hidden_state = torch.div(masked_hidden_state, num_tokens)     # batch x hidden_dim

        avg_masked_hidden_state = avg_masked_hidden_state.detach().cpu()

        return avg_masked_hidden_state / torch.norm(avg_masked_hidden_state)


if __name__ == '__main__':

    preferences = ["The preference is short, concise, academic writing", "The preference is bullet point",
                   "The preference is comic writing"]

    encoder = BertEncoding()
    v = encoder.encode(preferences)
