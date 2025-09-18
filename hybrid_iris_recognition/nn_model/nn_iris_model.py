import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tools.utils import manage_best_model_and_metrics


class nn_classifier_class():

    __slots__ = ['config', 'model', 'device', 'criterion', 'batch_size', 'optimizer', 'lower_is_better', 'best_metric']


    def __init__(self, model, config):
        self.config = config.training.nn

        if self.config.want_to_use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')           
        else:
            self.device = torch.device('cpu')
         
        self.model = model    
        self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.batch_size = self.config.batch_size
        self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=self.config.lr)

        if self.config.lower_is_better:
                self.best_metric = float('inf')
        else:
            self.best_metric = float('-inf')


    def fit(self, X, y):
        """
        Train the neural network model using the provided training data X and labels y.

        Args:
            X (numpy.ndarray): Training data of shape (num_samples, num_features).
            y (numpy.ndarray): Training labels of shape (num_samples,).
        """
        X_ten = torch.tensor(X, dtype=torch.float32) 
        y_ten = torch.tensor(y, dtype=torch.long) 
        dataset = torch.utils.data.TensorDataset(X_ten, y_ten)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        epochs = self.config.num_epochs
        total_steps = len(dataloader) * epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.config.lr, total_steps=total_steps)

        best_model = None
        lower_is_better = self.config.lower_is_better
        best_metric = self.best_metric

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            predictions = []
            references = []

            for batch in dataloader:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                scheduler.step()

                running_loss += loss.item()
                pred = torch.argmax(outputs, dim=1)
                predictions.extend(pred.cpu().numpy())
                references.extend(labels.cpu().numpy())

            metrics = {
                        'loss': running_loss / len(dataloader),
                        'accuracy': accuracy_score(references, predictions),
                        'precision': precision_score(references, predictions, average='macro', zero_division=1),
                        'recall': recall_score(references, predictions, average='macro', zero_division=1),
                        'f1_score': f1_score(references, predictions, average='macro', zero_division=1)
                        }
            best_model, best_metric = manage_best_model_and_metrics(self.model, 
                                                                    self.config.evaluation_metric,
                                                                    metrics, 
                                                                    best_metric, 
                                                                    best_model,
                                                                    lower_is_better
                                                                    )

        self.model = best_model  


    def predict(self, X):
        """
        Predict the class labels for the input data X.
        """
        X_ten = torch.tensor(X, dtype=torch.float32)  
        if X.shape[0] == 1:
            batch_size = 1
        else:
            batch_size = self.batch_size
        dataset = torch.utils.data.TensorDataset(X_ten)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            
            for batch in dataloader:
                inputs = batch[0]
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                pred = torch.argmax(outputs, dim=1)
                predictions.extend(pred.cpu().numpy())
        predictions_np_array = np.array(predictions)
        return predictions_np_array
    
    
    def load_weights(self, weights_path):
        if self.config.want_to_use_cuda and torch.cuda.is_available():
           self.model.load_state_dict(torch.load(weights_path, weights_only=True))
        else:
           self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True))
           