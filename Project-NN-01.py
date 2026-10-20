import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

os.makedirs("images", exist_ok=True)
# 1. Load, Inspect, and Preprocess the Dataset

def get_dataloaders(batch_size=64):
    transform = transforms.ToTensor()  # converts image to tensor, scales 0-255 -> 0-1

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset


def explore_data(train_dataset, test_dataset):
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    image, label = train_dataset[0]
    print(f"Image shape: {image.shape}")                       # [1, 28, 28]
    print(f"Pixel value range: {image.min():.2f} to {image.max():.2f}")  # 0.0 to 1.0
    print(f"Example label: {label}")

    # Visualize a grid of digits
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        img, lbl = train_dataset[i]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f"Label: {lbl}")
        ax.axis('off')
    plt.suptitle("Sample MNIST Digits")
    plt.tight_layout()
    plt.savefig("images/sample_digits.png")
    plt.close()
    print("Saved: sample_digits.png\n")


# 2. MLP Architecture

class MLP(nn.Module):
    def __init__(self, input_size=784,
                  hidden1=128, hidden2=32, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = self.flatten(x)       # (batch, 1, 28, 28) -> (batch, 784)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)           # (criterion applies softmax internally)
        return x


# 3. Training Loop (one epoch)

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()                     # 1. clear old gradients
        predictions = model(images)               # 2. forward pass
        loss = criterion(predictions, labels)     # 3. calculate loss
        loss.backward()                           # 4. compute gradients (chain rule)
        optimizer.step()                          # 5. update ALL weights

        total_loss += loss.item()
        _, predicted = torch.max(predictions, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train_model(model, train_loader, criterion, optimizer, device, epochs):
    history = {"loss": [], "accuracy": []}

    for epoch in range(epochs):
        avg_loss, accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Train Accuracy: {accuracy:.2f}%")

    return history


# 4. Evaluation

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():                           # disable gradient tracking (inference only)
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    return accuracy, all_preds, all_labels


def plot_loss(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history["loss"], marker='o')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("images/training_loss.png")
    plt.close()
    print("Saved: training_loss.png")


def plot_confusion_matrix(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("images/confusion_matrix.png")
    plt.close()
    print("Saved: confusion_matrix.png")


# 5. Error Analysis (Misclassified Images)

def show_misclassified(model, test_loader, device, num_images=10):
    model.eval()
    misclassified = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            wrong_mask = predicted != labels
            wrong_images = images[wrong_mask]
            wrong_preds = predicted[wrong_mask]
            wrong_labels = labels[wrong_mask]

            for img, pred, true in zip(wrong_images, wrong_preds, wrong_labels):
                misclassified.append((img.cpu(), pred.item(), true.item()))
                if len(misclassified) >= num_images:
                    break
            if len(misclassified) >= num_images:
                break

    if not misclassified:
        print("No misclassified images found!")
        return

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for ax, (img, pred, true) in zip(axes.flat, misclassified):
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f"True: {true}, Pred: {pred}")
        ax.axis('off')
    plt.suptitle("Misclassified Examples")
    plt.tight_layout()
    plt.savefig("images/misclassified.png")
    plt.close()
    print("Saved: misclassified.png")


# 6. Inference on a Single Image (using saved model)

def predict_single_image(model, test_dataset, device, index=0):
    model.eval()
    image, true_label = test_dataset[index]
    image_batch = image.unsqueeze(0).to(device)   # add batch dimension: (1, 1, 28, 28)

    with torch.no_grad():
        output = model(image_batch)
        predicted_label = torch.argmax(output, dim=1).item()

    print(f"\n[Inference Test] True label: {true_label} | Predicted label: {predicted_label}")

    plt.figure(figsize=(3, 3))
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"True: {true_label}, Predicted: {predicted_label}")
    plt.axis('off')
    plt.savefig("images/single_prediction.png")
    plt.close()
    print("Saved: single_prediction.png")


# ======================================================
# MAIN: Run Everything End-to-End
# ======================================================

if __name__ == "__main__":

    # ---- Configuration ----
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ---- Step 1 & 2: Load + Explore + Preprocess ----
    train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(BATCH_SIZE)
    explore_data(train_dataset, test_dataset)

    # ---- Step 3: Build Model ----
    model = MLP().to(device)
    print(model, "\n")

    # ---- Step 4: Configure Training ----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---- Step 5: Train ----
    history = train_model(model, train_loader, criterion, optimizer, device, EPOCHS)

    # ---- Step 6: Evaluate ----
    accuracy, all_preds, all_labels = evaluate_model(model, test_loader, device)
    plot_loss(history)
    plot_confusion_matrix(all_labels, all_preds)

    # ---- Step 7: Error Analysis ----
    show_misclassified(model, test_loader, device, num_images=10)

    # ---- Step 8: Save Model + Test Inference ----
    torch.save(model.state_dict(), "mnist_mlp.pth")
    print("\nModel saved as mnist_mlp.pth")

    predict_single_image(model, test_dataset, device, index=0)

    print("END")