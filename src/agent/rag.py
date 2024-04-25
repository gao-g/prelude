class RAG:
    def __init__(self, encode):
        self._encode = encode
        self._encodings = {}
        self.items = []

    def encode(self, doc):
        if doc not in self._encodings:
            self._encodings[doc] = self._encode(doc)
        return self._encodings[doc]

    def add(self, doc, whatever) -> None:
        self.items.append((self.encode(doc), doc, whatever))
        
    def get(self, doc, topk=1) -> list:
        from torch import matmul, stack
        topk = min(topk, len(self))
        if topk == 0:
            return []
        docs = stack([enc for enc, _, __ in self.items])
        sims = matmul(docs, self.encode(doc))
        return [self.items[i] for i in sims.topk(topk)[1]]
    
    def __len__(self):
        return len(self.items)