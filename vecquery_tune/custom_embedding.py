from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

class CustomEmbeddingModel(EmbeddingFunction):
    def __init__(self, model, tokenizer, device) -> None:
        super(CustomEmbeddingModel, self).__init__()
        self.embed = model
        self.tokenizer = tokenizer
        self.device = device
    def __call__(self, texts: Documents) -> Embeddings:
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            # put data on device
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            # get embeddings
            embedding = self.embed(input_ids, attention_mask)
            # put embeddings on cpu
            embedding = embedding.cpu()
            # convert embedding tensor to list
            embedding = embedding.detach()[0].numpy().tolist()
            # append embedding to embeddings list
            embeddings.append(embedding)
        return embeddings