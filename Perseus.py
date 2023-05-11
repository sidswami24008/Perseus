import torch
import os.path
import json
import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_layers, learning_rate, checkpoint_path, dictionary_path):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.checkpoint_path = checkpoint_path
        self.dictionary_path = dictionary_path
        self.model = self.build_model()
        self.dictionary = self.load_dictionary() if dictionary_path is not None else {}

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            self.load()

    def build_model(self):
        layers = []
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]

        for i in range(len(layer_sizes) - 1):
            layer = torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            layers.append(layer)
            layers.append(torch.nn.ReLU())

        layers.pop()
        model = torch.nn.Sequential(*layers)
        return model

    def forward(self, x):
        return self.model(x)

    def set_location(self):
        folder_location = "C:\\Users\\pokes\\PycharmProjects\\Perseus"
        essentials_location = os.path.join(folder_location, 'essentials')

        # create essentials folder if it doesn't exist
        if not os.path.exists(essentials_location):
            os.makedirs(essentials_location)

        # set checkpoint and dictionary file paths inside essentials folder
        self.checkpoint_path = os.path.join(essentials_location, 'checkpoint.pt')
        self.dictionary_path = os.path.join(essentials_location, 'dictionary.json')

        # create checkpoint file if it doesn't exist
        if not os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'w') as f:
                f.write('checkpoint file created')

        # create dictionary file if it doesn't exist
        if not os.path.exists(self.dictionary_path):
            with open(self.dictionary_path, 'w') as f:
                f.write('dictionary file created')

    def save(self):
        checkpoint = {'input_size': self.input_size,
                      'output_size': self.output_size,
                      'hidden_layers': self.hidden_layers,
                      'learning_rate': self.learning_rate,
                      'state_dict': self.model.state_dict()}
        torch.save(checkpoint, self.checkpoint_path)

    def load(self):
        checkpoint = torch.load(self.checkpoint_path)
        self.input_size = checkpoint['input_size']
        self.output_size = checkpoint['output_size']
        self.hidden_layers = checkpoint['hidden_layers']
        self.learning_rate = checkpoint['learning_rate']
        self.model.load_state_dict(checkpoint['state_dict'])

        if not os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'w'):
                pass

        if not os.path.exists(self.dictionary_path):
            with open(self.dictionary_path, 'w'):
                pass

    def load_dictionary(self):
        with open(self.dictionary_path, 'r') as f:
            dictionary = json.load(f)
        return dictionary

    def save_dictionary(self):
        with open(self.dictionary_path, 'w') as f:
            json.dump(self.dictionary, f)

    def edit_weights(self, layer, row, col, value):
        self.model[layer].weight[row][col] = value

    def edit_biases(self, layer, index, value):
        self.model[layer].bias[index] = value

    def add_layer(self, size):
        self.model = torch.nn.Sequential(self.model, torch.nn.Linear(self.hidden_layers[-1], size), torch.nn.ReLU())
        self.hidden_layers.append(size)

    def remove_layer(self):
        self.model = torch.nn.Sequential(*list(self.model.children())[:-2])
        self.hidden_layers.pop()

    def train(self, inputs, labels, data_type='image'):
        inputs = torch.from_numpy(np.array(inputs)).float()
        labels = torch.from_numpy(np.array(labels)).long()

        if data_type == 'image':
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

            for epoch in range(100):
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        elif data_type == 'text':
            # Assume inputs are already preprocessed with word embeddings
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

            for epoch in range(100):
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Save word "training" to dictionary
            if self.dictionary_path is not None:
                with open(self.dictionary_path, 'r') as f:
                    dictionary = json.load(f)
                dictionary['training'] = True
                with open(self.dictionary_path, 'w') as f:
                    json.dump(dictionary, f)

        elif data_type == 'both':
            image_inputs, text_inputs = inputs
            image_inputs = torch.from_numpy(np.array(image_inputs)).float()
            text_inputs = torch.from_numpy(np.array(text_inputs)).float()

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

            for epoch in range(100):
                optimizer.zero_grad()
                outputs = self.forward(image_inputs)
                outputs += self.forward(text_inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Save word "training" to dictionary
            if self.dictionary_path is not None:
                with open(self.dictionary_path, 'r') as f:
                    dictionary = json.load(f)
                dictionary['training'] = True
                with open(self.dictionary_path, 'w') as f:
                    json.dump(dictionary, f)

        if self.checkpoint_path is not None:
            self.save()
            optimizer.zero_grad()
            outputs = self.forward(image_inputs)
            outputs += self.forward(text_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Save word "training" to dictionary
            if self.dictionary_path is not None:
                with open(self.dictionary_path, 'r') as f:
                    dictionary = json.load(f)
                dictionary['training'] = True
                with open(self.dictionary_path, 'w') as f:
                    json.dump(dictionary, f)

            if self.checkpoint_path is not None:
                self.save()