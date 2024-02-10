import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

TOKENIZER = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def score_model(sentences, tokenizer, model):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

def main():
    df = pd.read_csv("city_wikipedia_summaries.csv")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    model = AutoModel.from_pretrained(MODEL)
    embeddings = score_model(df['Wiki Summary'].tolist(), tokenizer, model)
    print(embeddings)
    print('shape = ', df.shape)
    df['embeddings'] = list(embeddings.detach().cpu().numpy())
    print(df.head())

if __name__ == '__main__':
    main()
