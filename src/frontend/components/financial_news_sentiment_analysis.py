from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import streamlit as st

import os
import torch
from torchtext.data.utils import get_tokenizer
import torch.nn as nn
import re
import nltk
import unidecode
import pickle

nltk.download('stopwords')

classes = {
    0: 'neutral',
    1: 'negative',
    2: 'positive'
}

device = torch.device('cpu')


class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim,
                 hidden_size, n_layers, n_classes,
                 dropout_prob):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size,
                          n_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, hn = self.rnn(x)
        x = x[:, -1, :]
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Load vocab
vocab_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../weights/vocabulary.pkl'))
with open(vocab_path, 'rb') as f:
    vocab_list = pickle.load(f)
    vocab = {word: idx for idx, word in enumerate(vocab_list)}

vocab_size = len(vocab)
n_classes = len(list(classes.keys()))
embedding_dim = 64
hidden_size = 64
n_layers = 2
dropout_prob = 0.2


@st.cache_resource
def load_model(model_path):
    model = SentimentClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        n_layers=n_layers,
        n_classes=n_classes,
        dropout_prob=dropout_prob
    )
    model.load_state_dict(torch.load(
        model_path, weights_only=True, map_location=device))
    model.eval()
    model.dropout.train(False)
    return model


model_path = absolute_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../weights/sentiment_analysis_model.pt'))
model = load_model(model_path)

tokenizer = get_tokenizer('basic_english')
english_stop_words = stopwords.words('english')
stemmer = PorterStemmer()


def text_normalize(text):
    text = text.lower()
    text = unidecode.unidecode(text)
    text = text.strip()
    text = re.sub(r'[~\w\s]', '', text)
    text = ' '.join([word for word in text.split(
        ' ') if word not in english_stop_words])
    text = ' '.join([stemmer.stem(word) for word in text.split(' ')])
    return text


def inference(text, model):
    normalized_text = text_normalize(text)
    tokens = tokenizer(normalized_text)

    max_seq_len = 25
    indices = []
    for token in tokens:
        if token in vocab:
            indices.append(vocab[token])
        else:
            indices.append(vocab['UNK'])

    # Pad or truncate sequence
    if len(indices) < max_seq_len:
        indices += [vocab['PAD']] * (max_seq_len - len(indices))
    else:
        indices = indices[:max_seq_len]

    text_tensor = torch.LongTensor(indices).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(text_tensor)
    probabilities = nn.Softmax(dim=1)(outputs)
    p_max, y_hat = torch.max(probabilities, dim=1)

    return p_max.item(), y_hat.item()


def run():
    st.title('Financial News Sentiment Analysis')
    st.header('Input News')
    financial_news = st.text_area(
        'Enter your financial news text here:',
        value='With the new production plant the company would increase its capacity to meet the expected increase in demand and would improve the use of raw materials and therefore increase the production profitability.',
        height=340
    )
    if st.button('Analyze Sentiment'):
        p_max, y_hat = inference(financial_news, model)
        sentiment = classes[y_hat]
        st.success(
            f"Sentiment: **{sentiment}** with **{p_max:.2%}** probability")


if __name__ == '__main__':
    run()
