import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os
import json
from torch import nn
import torch.optim as optim
import random
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import ssl
from multiprocessing import freeze_support

ssl._create_default_https_context = ssl._create_unverified_context

globals()


def main():
    checkpoint_path = 'checkpoint.pt'  # Specify the checkpoint file path
    dictionary_path = 'dictionary.json'  # Specify the dictionary file path
    cifar10_batch_size = 32  # Specify the batch size for CIFAR-10
    zest_batch_size = 32  # Specify the batch size for ZEST
    network.load_zest_train_data_loader(zest_batch_size)
    network.load_zest_test_data_loader(zest_batch_size)
    network.load_cifar10_data_loader(cifar10_batch_size)
    network.train_zest_model(zest_batch_size=zest_batch_size, num_epochs=1)
    network.train_cifar10_model(cifar10_batch_size=cifar10_batch_size, num_epochs=1)

    network.train_and_evaluate(cifar10_batch_size=cifar10_batch_size, zest_batch_size=zest_batch_size, num_epochs=1)


def preprocess_text(text):
    return text.lower()


def preprocess_data(df):
    if 'text' not in df.columns:
        df['text'] = ''  # Create 'text' column with empty values

    df['text'] = df['text'].apply(preprocess_text)  # Apply preprocessing function to 'text' column

    return df


def tokenize_data(data):
    # Tokenize the preprocessed data using your preferred method
    vectorizer = CountVectorizer()  # Example: Using CountVectorizer for tokenization
    tokenized_data = vectorizer.fit_transform(data)
    return tokenized_data.toarray()  # Convert to dense matrix


def create_cifar10_data_loader(images, labels, batch_size, shuffle=True):
    if isinstance(images, np.ndarray):  # For CIFAR-10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the data
        ])
        dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader
    else:  # For ZEST
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(images.astype('float32')))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader


def create_zest_data_loader(tokenized_data, batch_size, shuffle=True):
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(tokenized_data.astype('float32')))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


dataset = "cifar10"  # or "zest"
if dataset == "cifar10":
    input_size = 3 * 32 * 32  # CIFAR-10 input size
    output_size = 10  # CIFAR-10 output size
elif dataset == "zest":
    input_size = 1 * 64 * 64  # ZEST input size
    output_size = 2  # ZEST output size
else:
    raise ValueError("Unsupported dataset: " + dataset)


class PerseusNetwork:
    def __init__(self, input_size, output_size, checkpoint_path=None, dictionary_path=None):
        super(PerseusNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(self.input_size, self.output_size)
        self.hidden_layers = 1  # Set the default value for hidden_layers
        self.learning_rate = random.uniform(0.001, 0.1)
        self.neurons = []
        self.neural_connections = []
        self.check_files(checkpoint_path, dictionary_path)
        self.checkpoint_path = checkpoint_path
        self.dictionary_path = dictionary_path
        self.model = self.build_model()
        self.train_loader = None
        self.test_loader = None
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            self.load_model()

        if dictionary_path is not None and os.path.exists(dictionary_path):
            self.load_dictionary()

    def forward(self, x):
        print("Input shape:", x.shape)
        x = self.linear(x)
        return x

    @staticmethod
    def check_files(checkpoint_path, dictionary_path):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoint.pt')
        if dictionary_path is None:
            dictionary_path = os.path.join(os.path.dirname(__file__), 'dictionary.json')

        if not os.path.exists(checkpoint_path):
            open(checkpoint_path, 'a').close()

        if not os.path.exists(dictionary_path):
            open(dictionary_path, 'a').close()

    def set_input_size(self, size):
        self.input_size = size

    def set_output_size(self, size):
        self.output_size = size

    def set_hidden_layers(self, layers):
        try:
            self.hidden_layers = int(layers)
        except ValueError:
            raise ValueError("The number of hidden layers must be an integer.")
        self.neurons = [random.randint(1, 100) for _ in range(self.hidden_layers)]
        self.neural_connections = [[[random.uniform(-1, 1) for _ in range(size)] for _ in range(size)] for size in
                                   self.neurons]

    def set_learning_rate(self, rate):
        self.learning_rate = rate

    def set_neuron(self, layer, value):
        if 0 <= layer < len(self.neurons):
            self.neurons[layer] = value

    def set_neural_connection(self, layer, row, col, value):
        if 0 <= layer < len(self.neural_connections) and 0 <= row < len(
                self.neural_connections[layer]) and 0 <= col < len(self.neural_connections[layer][row]):
            self.neural_connections[layer][row][col] = value

    def build_model(self):
        layers = []
        layer_sizes = [self.input_size] + [self.hidden_layers] + [self.output_size]

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            layers.append(layer)
            layers.append(nn.ReLU())

        layers.pop()
        model = nn.Sequential(*layers)
        return model

    def load_model(self):
        checkpoint = torch.load(self.checkpoint_path)
        self.input_size = checkpoint['input_size']
        self.output_size = checkpoint['output_size']
        self.hidden_layers = checkpoint['hidden_layers']
        self.learning_rate = checkpoint['learning_rate']
        self.model = self.build_model()
        self.model.load_state_dict(checkpoint['state_dict'])

    def save_model(self):
        checkpoint = {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'hidden_layers': self.hidden_layers,
            'learning_rate': self.learning_rate,
            'state_dict': self.model.state_dict()
        }
        torch.save(checkpoint, self.checkpoint_path)

    def save_model_checkpoint(self, checkpoint_path):
        torch.save(self.model.state_dict(), checkpoint_path)

    def load_model_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()

    def load_dictionary(self, dictionary_path):
        with open(dictionary_path, 'r') as f:
            self.dictionary = json.load(f)

    def save_dictionary(self):
        with open(self.dictionary_path, 'w') as f:
            json.dump(self.dictionary, f)

    def predict_text(self, text):
        # Preprocess the input text
        tokens = text.lower().split()
        tokens = [self.dictionary.get(token, self.dictionary['<unk>']) for token in tokens]
        input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

        # Pass the input tensor through the model
        output = self.model(input_tensor)

        # Get the predicted class
        _, predicted = torch.max(output.data, 1)

        # Convert the predicted class to the corresponding text label
        label = self.labels[predicted.item()]

        return label

    def load_cifar10_data_loader(self, cifar10_batch_size):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the data
        ])

        # Update the root parameter to the desired directory path
        trainset = torchvision.datasets.CIFAR10(root='C:/Users/pokes/PycharmProjects/Perseus/data',
                                                train=True, download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=cifar10_batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='C:/Users/pokes/PycharmProjects/Perseus/data',
                                               train=False, download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=cifar10_batch_size, shuffle=False, num_workers=2)
        self.train_loader = trainloader
        self.test_loader = testloader
        return trainloader, testloader

    def load_cifar10_train_data_loader(self, cifar10_batch_size):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the data
        ])

        train_data_files = [f"data_batch_{i}" for i in range(1, 6)]
        train_data = []
        for file_name in train_data_files:
            trainset = torchvision.datasets.CIFAR10(
                root='/data/cifar-10-batches-py',
                train=True,
                download=False,
                transform=transform
            )
            train_data.extend(trainset)

        cifar10_train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=cifar10_batch_size,
            shuffle=True,
            num_workers=2
        )

        return cifar10_train_loader

    def load_cifar10_test_data_loader(self, cifar10_batch_size):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the data
        ])

        testset = torchvision.datasets.CIFAR10(
            root='/data/cifar-10-batches-py',
            train=False,
            download=False,
            transform=transform
        )
        cifar10_test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=cifar10_batch_size,
            shuffle=False,
            num_workers=2
        )

        return cifar10_test_loader

    def train_cifar10_model(self, num_epochs, cifar10_batch_size):
        if self.train_loader is None:
            print("CIFAR-10 training data not loaded. Please load the data before training.")
            return
        self.load_cifar10_train_data_loader(cifar10_batch_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        running_loss = 0.0
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(self.train_loader, 0):
                # Rest of the training code

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                images = images.to(self.device)  # Move images to the device (GPU or CPU)
                labels = labels.to(self.device)  # Move labels to the device (GPU or CPU)

                outputs = self.model(images.view(images.size(0), -1))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item()
                if i % 100 == 99:  # Print every 100 mini-batches
                    print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}")
                    running_loss = 0.0

        print("Finished training on CIFAR-10")

    def test_cifar10_model(self, cifar10_batch_size):
        if self.test_loader is None:
            print("CIFAR-10 testing data not loaded. Please load the data before testing.")
            return
        self.load_cifar10_test_data_loader(cifar10_batch_size)
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy on CIFAR-10: {accuracy:.2f}%")

    def load_zest_data_loader(self, file_path):
        file_path= 'data/zest'
        try:
            with open(file_path, 'r') as file:
                df = pd.read_json(file, lines=True)
            preprocessed_data = preprocess_data(df)
            tokenized_data = tokenize_data(preprocessed_data)
            return tokenized_data
        except FileNotFoundError:
            print(f"File not found at path: {file_path}")
        except Exception as e:
            print(f"An error occurred while loading data: {e}")

    def load_zest_train_data_loader(self, zest_batch_size):
        train_path = os.path.join(os.getcwd(), 'data/zest', 'train.jsonl')
        train_data = self.load_zest_data_loader(train_path)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data))
        self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=zest_batch_size, shuffle=True)

    def load_zest_test_data_loader(self, zest_batch_size):
        test_path = os.path.join(os.getcwd(), 'data/zest', 'test_unanswered.jsonl')
        test_data = self.load_zest_data_loader(test_path)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data))
        self.test_loader = torch.utils.data.DataLoader(dataset, batch_size=zest_batch_size, shuffle=False)

    def train_zest_model(self, zest_batch_size, num_epochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                inputs = data[0]
                labels = data[1]

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:
                    print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 2000}')
                    running_loss = 0.0

    def test_zest_model(self, zest_batch_size):

        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                images = data[0]
                labels = data[1]
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy on ZEST: {accuracy}%")

    def train_and_evaluate(self, num_epochs, cifar10_batch_size, zest_batch_size):
        self.load_cifar10_data_loader(cifar10_batch_size)
        self.load_zest_train_data_loader(zest_batch_size)
        # Training code
        for epoch in range(num_epochs):
            # Modify the parameters during training
            if epoch % 5 == 0:
                # Modify parameters for CIFAR-10
                self.set_input_size(random.randint(1, 100))
                self.set_output_size(random.randint(1, 100))
                self.set_hidden_layers(random.randint(1, 10))
                self.set_learning_rate(random.uniform(0.001, 0.1))

                # Modify parameters for ZEST
                for layer in range(self.hidden_layers):
                    self.set_neuron(layer, random.randint(1, 100))
                    for row in range(len(self.neural_connections[layer])):
                        for col in range(len(self.neural_connections[layer][row])):
                            self.set_neural_connection(layer, row, col, random.uniform(-1, 1))
                self.set_learning_rate(random.uniform(0.001, 0.1))

            # Train on CIFAR-10 data
            self.train_cifar10_model(cifar10_batch_size)

            # Train on ZEST data
            self.train_zest_model(zest_batch_size)

            # Evaluate the models
            self.test_cifar10_model(cifar10_batch_size)
            self.test_zest_model(zest_batch_size)


network = PerseusNetwork(input_size=input_size, output_size=output_size)  # Provide the output_size argument
if __name__ == '__main__':
    freeze_support()
    main()
