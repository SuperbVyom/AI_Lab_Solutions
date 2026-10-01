"""PyTorch XOR and three-class sensor experiments (CPU, deterministic seed)."""
import torch
from torch import nn

torch.set_num_threads(1)
X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
Y = torch.tensor([[0.],[1.],[1.],[0.]])
YC = torch.tensor([0,1,1,2])


class Net(nn.Module):
    def __init__(self, activation="tanh", outputs=1):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(2, 2), nn.Linear(2, outputs)
        self.activation = {"sigmoid": torch.sigmoid, "tanh": torch.tanh, "relu": torch.relu}[activation]
    def forward(self, x): return self.fc2(self.activation(self.fc1(x)))


def binary(activation="tanh", zero=False, steps=5000, seed=7):
    torch.manual_seed(seed); model = Net(activation)
    if zero:
        with torch.no_grad():
            for p in model.parameters(): p.zero_()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss(); initial = loss_fn(model(X), Y).item()
    early_grad = None
    for step in range(steps):
        opt.zero_grad(); loss = loss_fn(model(X), Y); loss.backward()
        if step == 0: early_grad = model.fc1.weight.grad.norm().item()
        opt.step()
    with torch.no_grad():
        probs = model(X).sigmoid().flatten(); pred = (probs >= .5).int()
    return initial, loss.item(), probs.tolist(), pred.tolist(), early_grad, model


def multiclass(steps=5000, seed=7):
    torch.manual_seed(seed); model = Net("tanh", outputs=3)
    opt = torch.optim.Adam(model.parameters(), lr=0.05); loss_fn = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad(); loss = loss_fn(model(X), YC); loss.backward(); opt.step()
    with torch.no_grad():
        logits = model(X); probs = logits.softmax(dim=1)
        shifted = (logits[0] + 100).softmax(dim=0)
    return loss.item(), probs.tolist(), probs[0].sum().item(), (shifted-probs[0]).abs().max().item(), model


if __name__ == "__main__":
    print("binary", binary()[:5])
    print("zero-init", binary(zero=True)[:5])
    for act in ("sigmoid", "tanh", "relu"):
        result = binary(act)
        print("activation", act, result[:5])
    print("multiclass", multiclass()[:4])
