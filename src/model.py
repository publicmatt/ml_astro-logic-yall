from torch import nn

class AstroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(512, 64),
            nn.Linear(64, 13)
        )
    def forward(self, x):
        return self.layers(x)
