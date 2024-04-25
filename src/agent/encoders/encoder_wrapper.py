from src.agent.encoders.bert import BertEncoding
from src.agent.encoders.mpnet_base import MPNetEncoding


class EncoderWrapper:

    def __init__(self):
        pass

    @staticmethod
    def make_encoder(enc_name):

        if enc_name == "bert":
            model = BertEncoding()

        elif enc_name == "mpnet":
            model = MPNetEncoding()

        else:
            raise AssertionError(f"Encoder name {enc_name} not implemented.")

        return model
