import torch                   
import torch.nn as nn          
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.datasets import fetch_openml
from torch.utils.data import TensorDataset, DataLoader


class Mnistnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__() 
        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out

def load_pictures(path):
    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), 
    transforms.Resize((28, 28)),                
    transforms.ToTensor()                        
    ])

    X_list = []
    y_list = []
    train_dataset = datasets.ImageFolder(root=path, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128,      
        shuffle=False, 
        num_workers=4,        
        pin_memory=True)


    for images, labels in train_loader:
        images = images.reshape(images.size(0), -1)
        X_list.append(images)
        y_list.append(labels)


    xtrain = torch.cat(X_list, dim=0)
    ytrain = torch.cat(y_list, dim=0)
    return xtrain,ytrain

def create_dataset(path, batch_size):
    xtrain, ytrain = load_pictures(path)
    fast_dataset = TensorDataset(xtrain, ytrain)
    fast_loader = DataLoader(fast_dataset, batch_size=batch_size, shuffle=True)

    return fast_loader

def train(path, epoch, model, batch_size,learning_rate):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    fast_loader = create_dataset(path, batch_size)

    for i in range(epoch):
        ep_loss=0
        for images,labels in fast_loader:
            preds = model(images)   
            loss = criterion(preds, labels)
            ep_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (i + 1) % 1 == 0:
            print(f"Epoch [{i+1}/{epoch}], Loss: {ep_loss/len(fast_loader)}")

def test(path, model):
    loader = create_dataset(path, 128)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for inputs,targets in loader:
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)


            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)
    avg_test_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    print(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {accuracy:.4f}")





envelope = Mnistnn(input_size=784, hidden_size=124, output_size=10)

train("./mnist-png/train", 30, envelope,128 ,0.001)
test("./mnist-png/test", envelope)