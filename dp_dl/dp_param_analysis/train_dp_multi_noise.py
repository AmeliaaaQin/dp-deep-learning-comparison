# ======================
# 0. Imports
# ======================
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from opacus import PrivacyEngine


# ======================
# 1. Model Definition
# ======================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.fc = nn.Linear(21632, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ======================
# 2. Dataset & Dataloader
# ======================
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# ======================
# 3. Training Utilities
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()


def train(model, loader, optimizer):
    model.train()
    total_loss = 0.0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def test(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    return correct / total


# ======================
# 4. Experiment Settings
# ======================
noise_list = [0.5, 0.8, 1.0, 1.2, 1.5]
epochs = 5
delta = 1e-5


# ======================
# 5. Main Experiment Loop
# ======================
results = []

print("===== DP-SGD Multi-Noise Experiment Start =====")

for noise in noise_list:
    print(f"\n--- Training with noise_multiplier = {noise} ---")

    # Reinitialize model and optimizer for each noise
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader_dp = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise,
        max_grad_norm=1.0,
    )

    for epoch in range(epochs):
        train_loss = train(model, train_loader_dp, optimizer)
        test_acc = test(model, test_loader)
        epsilon = privacy_engine.get_epsilon(delta=delta)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Accuracy: {test_acc:.4f} | "
            f"Epsilon: {epsilon:.2f}"
        )

    results.append({
        "noise": noise,
        "accuracy": test_acc,
        "epsilon": epsilon
    })


# ======================
# 6. Final Results Summary
# ======================
print("\n===== Final Results Summary =====")
for r in results:
    print(
        f"Noise: {r['noise']} | "
        f"Accuracy: {r['accuracy']:.4f} | "
        f"Epsilon: {r['epsilon']:.2f}"
    )
