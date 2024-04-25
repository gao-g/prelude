class AbstractEncoder:

    def __init__(self):
        pass

    def encode(self, text):
        """
        :param text: the text in string that needs to be encoded
        :return: a single tensor 1-d that contains encoding of the text
        """
        raise NotImplementedError()
