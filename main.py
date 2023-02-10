from datetime import datetime
import pandas as pd
import requests
from lxml.html import fromstring
import torch
from transformers import AutoTokenizer, BertModel, BertConfig
from pathlib import Path
import click
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

@click.group()
def cli():
    ...

@cli.command()
def data():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    df = pd.read_csv('data/famous-birthdates.csv', delimiter="|")
    base_dir = Path('/home/jensen33/Dev/astro-logic-all')

    tokens = []
    segments = []
    for name in df.name:
        url = f"https://en.wikipedia.org/w/index.php?search={name}"
        response = requests.get(url)
        tree = fromstring(response.content)
        description = tree.xpath("//div[@id='bodyContent']")[0].text_content()
        marked_text = "[CLS] " + description + " [SEP]"
        # tokenized_text = tokenizer.tokenize(marked_text)
        tokenized_text = tokenizer.tokenize(marked_text, truncation=True, max_length=512)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        tokens.append(tokens_tensor)
        segments.append(segments_tensors)

    def save_squeezed(ten, filename):
        squeezed = [x.squeeze() for x in ten]
        padded = pad_sequence(squeezed, batch_first=False, padding_value=-1)
        torch.save(padded, base_dir / 'data' / filename)

    save_squeezed(tokens, 'tokens.pt')
    save_squeezed(segments, 'segments.pt')
    return tokens

@cli.command()
def train():
    tokens = torch.load(base_dir / 'data/tokens.pt')
    #segments = torch.load(base_dir / 'data/tokens.pt')

torch.unsqueeze(tokens, dim=0).shape

x = trainset.x
y = trainset.y

import sklearn
from sklearn.neural_network import MLPClassifier

X = trainset.x
y = trainset.y
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)

name = "Mitch Hedberg"

def get_wiki_tokens(name):
    url = f"https://en.wikipedia.org/w/index.php?search={name}"
    response = requests.get(url)
    tree = fromstring(response.content)
    description = tree.xpath("//div[@id='bodyContent']")[0].text_content()
    marked_text = "[CLS] " + description + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text, truncation=True, max_length=512)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    return indexed_tokens

tokenized_text
x = get_wiki_tokens("Artie Lange")
pred = clf.predict([x]) 
list(mapped.keys())[list(mapped.values()).index(pred)]
x.shape


class AstroDataset(Dataset):
    base_dir = Path('/home/jensen33/Dev/astro-logic-all')
    def __init__(self, feature_path, target_path):
        x = torch.load(base_dir / feature_path)
        self.x = torch.permute(x, (1, 0))
        df = pd.read_csv(base_dir / target_path, delimiter="|")
        signs = df.zodiac.unique().tolist()
        mapped = dict(zip(signs, range(len(signs))))
        self.y = torch.tensor(df.zodiac.apply(lambda x: mapped[x]).tolist())
    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])
    def __len__(self):
        return len(self.x)

class AstroModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(512, 64),
            torch.nn.Linear(64, 13)
        )
    def forward(self, x):
        return self.layers(x)

trainset = AstroDataset('data/tokens.pt', 'data/famous-birthdates.csv')

batch_size = 16
epochs = 16
lr = 1e-3
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
model = AstroModel()
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
model.train()

loss = 0
for epoch in range(epochs):
    for batch, (x, y) in enumerate(trainloader):
        optimizer.zero_grad()
        pred_y = model(x)
        loss = loss_function(pred_y, y)
        loss_function.backward(loss)
        optimizer.step()  # Perform a gradient update on the weights of the mode
        loss += loss.item()


model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
model.eval()



signs = [
    {'sign': 'Aries', 'start': 'March 21', 'end': 'April 19'},
    {'sign': 'Taurus', 'start': 'April 20', 'end': 'May 20'},
    {'sign': 'Gemini', 'start': 'May 2', 'end':'June 21'},
    {'sign': 'Cancer', 'start': 'June 2', 'end':'July 22'},
    {'sign': 'Leo', 'start': 'July 23', 'end': 'August 22'},
    {'sign': 'Virgo', 'start': 'August 23', 'end': 'September 22'},
    {'sign': 'Libra', 'start': 'September 23', 'end': 'October 23'},
    {'sign': 'Scorpio', 'start': 'October 24', 'end': 'November 21'},
    {'sign': 'Sagittarius', 'start': 'November 22', 'end': 'December 21'},
    {'sign': 'Capricorn', 'start': 'December 22', 'end': 'January 19'},
    {'sign': 'Aquarius', 'start': 'January 20', 'end': 'February 18'},
    {'sign': 'Pisces', 'start': 'February 19', 'end': 'March 20'}
]

