import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from opacus import PrivacyEngine
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# 模型定义
# ======================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# ======================
# 数据加载函数
# ======================
def get_data_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_set = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)

    return train_loader, test_loader

# ======================
# 训练函数
# ======================
def train_dp(noise_multiplier, batch_size, epochs=10):

    train_loader, test_loader = get_data_loaders(batch_size)

    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=1.0,
    )

    delta = 1e-5
    epsilon_list = []

    for epoch in range(epochs):
        model.train()

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        epsilon = privacy_engine.get_epsilon(delta)
        epsilon_list.append(epsilon)

        print(f"Epoch {epoch+1}/{epochs} | Epsilon: {epsilon:.4f}")

    # 测试准确率
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    accuracy = correct / total

    return accuracy, epsilon_list


# ======================
# 主实验流程
# ======================
if __name__ == "__main__":

    noise_list = [0.5, 1.0, 1.5]
    batch_size_list = [32, 64]

    results = {}

    for batch_size in batch_size_list:
        for noise in noise_list:

            print(f"\nRunning | Batch: {batch_size} | Noise: {noise}")

            acc, eps_list = train_dp(noise, batch_size)

            results[(batch_size, noise)] = {
                "accuracy": acc,
                "epsilon_curve": eps_list
            }

            print(f"Final Accuracy: {acc:.4f}")
            print(f"Final Epsilon: {eps_list[-1]:.4f}")

    # 示例：画一个 epsilon 曲线
    sample_key = list(results.keys())[0]
    plt.plot(results[sample_key]["epsilon_curve"])
    plt.title("Epsilon vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Epsilon")
    plt.show()
