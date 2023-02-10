import torch
import click
import sklearn
from sklearn.neural_network import MLPClassifier
from dataset import AstroDataset, get_wiki_tokens
from model import AstroModel

@cli.command()
def train():
    batch_size = 16
    epochs = 16
    lr = 1e-3
    trainset = AstroDataset('data/tokens.pt', 'data/famous-birthdates.csv')
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

    X = trainset.x
    y = trainset.y
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X, y)
    x = get_wiki_tokens("Artie Lange")
    mapped = trainset.mapped

def predict():
    name = "Mitch Hedberg"
    pred = clf.predict([x]) 
    list(mapped.keys())[list(mapped.values()).index(pred)]

@click.group()
def cli():
    ...

if __name__ = "__main__":
    cli()
