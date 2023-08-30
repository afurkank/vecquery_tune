from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

class CustomEmbeddingModel(EmbeddingFunction):
    def __init__(self, model, tokenizer) -> None:
        super(CustomEmbeddingModel, self).__init__()
        self.embed = model
        self.tokenizer = tokenizer
    def __call__(self, texts: Documents) -> Embeddings:
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            embedding = self.embed(input_ids, attention_mask).detach()[0].numpy().tolist()
            embeddings.append(embedding)
        return embeddings