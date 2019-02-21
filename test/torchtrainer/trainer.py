import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import os
import time
from .utils import Progbar

class Trainer(object):
    """Class abstracting model training in PyTorch."""
    
    def __init__(self, model, loss_fn, optimizer, metrics=None, device="auto"):
        """Wrapper class to train a PyTorch model.
        Args:
            model: A Pytorch model.
            loss_fn: A function returning the loss.
            optimizer: A Pytorch optimizer (torch.optim).
            device: Can be either a torch.device instance or one of the choices:
                "auto", "cpu", "cuda".
        """
        assert device in ("auto", "cpu", "cuda") or isinstance(device, torch.device)
        if device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif device == "cpu":
            self.device = torch.device("cpu")
        elif device == "cuda":
            self.device = torch.device("cuda")
        elif isinstance(device, torch.device):
            self.device = device
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics
        
        self.best_val_loss = np.inf
        
    def fit(self, train_loader, val_loader=None, epochs=1, checkpoint_path=None, 
            early_stopping=False, verbose=1, plot_loss=False):
        """Train the model.
        Args:
            train_loader: A Pytorch Dataloader returning the training batches.
            val_loader: A Pytorch Dataloader returning the validation batches.
            epochs (int): The number of iteration for training.
            checkpoint_path: The path to the file where a checkpoint will be 
                saved when the model improve (a val_loader is necessary).
            early_stopping (int or None): If a number n is specified, 
                early stopping will be done if there is no improvement on the 
                validation set after n epochs.
            verbose: Verbosity of the progress bar: 
                0 (almost silent), 1 (verbose), 2 (semi-verbose).
            plot_loss (bool): If True, plot the training and validation loss 
            at the end.
        """
                    
        if early_stopping or checkpoint_path is not None:
            assert val_loader is not None
            if early_stopping:
                assert isinstance(early_stopping, int)
            n = 0
            
        if plot_loss:
            logs = {}
        
        for epoch in range(1, epochs + 1):
            
            print("Epoch {}/{}".format(epoch, epochs))
            self.progbar = self._make_progbar(train_loader, val_loader, verbose)
            
            # Training loop
            self._train_loop(train_loader)
            
            # Validation loop
            if val_loader:
                self._validate_loop(val_loader)
            
            # Get average value of all metrics for last epoch
            metrics_val = self.progbar.logger.average()
            
            if val_loader:
                # Save best model if improvement on validation loss
                if checkpoint_path and metrics_val["val_loss"] < self.best_val_loss:
                    # Save model_dict model_state_dict, optimizer_state_dict
                    # and all metrics in progbar.
                    self.save_checkpoint(checkpoint_path, metrics_val)
                    print("Model improved, saved at " + checkpoint_path)
                
                # Check for early stopping
                if early_stopping:
                    if metrics_val["val_loss"] >= self.best_val_loss:
                        n += 1
                        if n < early_stopping:
                            print("No improvement in %d Epochs." % n)
                        if n >= early_stopping:
                            print("No improvement in %d Epochs: Early Stopping." % n)
                            break
                    else:
                        n = 0
                
                # Update best_val_loss
                if metrics_val["val_loss"] < self.best_val_loss:
                    self.best_val_loss = metrics_val["val_loss"]
                    
            if plot_loss:
                for key in metrics_val:
                    if key not in logs:
                        logs[key] = [metrics_val[key]]
                    else:
                        logs[key].append(metrics_val[key])
        
        # Plot loss
        if plot_loss:
            plt.figure()
            plt.plot(logs["train_loss"], label="Training")
            if val_loader:
                plt.plot(logs["val_loss"], label="Validation")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.legend()
            plt.show()
    
    def train(self, batch):
        """A single step of training through one batch. 
        Can be overwritten, useful if batch is a dictionnary, for example.
        Args:
            batch: A single batch returned by a Dataloader.
        Return:
            A list of tuple, each tuple must contain a key (string) and a value 
            (float or int), to track on the progress bar.
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Track metrics
        values = {"train_loss": loss.item()}
        for k in self.metrics:
            values["train_" + k] = self.metrics[k](y_pred, y)
        return values
    
    def validate(self, batch):
        """A single step of validation through one batch. 
        Can be overwritten, useful if batch is a dictionnary, for example.
        Args:
            batch: A single batch returned by a Dataloader.
        Return:
            A list of tuple, each tuple must contain a key (string) and a value 
            (float or int), to track on the progress bar.
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)

         # Track metrics
        values = {"val_loss": loss.item()}
        for k in self.metrics:
            values["val_" + k] = self.metrics[k](y_pred, y)
        return values
    
    def _train_loop(self, train_loader):
        """Do a single epoch of training."""
        # Set model to train mode
        self.model.train()
        
        for batch in train_loader:
                
            values = self.train(batch)
                
            # Update progress bar
            self.progbar.update(values)
            
    def _validate_loop(self, val_loader):
        """Do a single epoch of validating."""
        # Set model to eval mode
        self.model.eval()
        
        with torch.no_grad():
            for batch in val_loader:
                values = self.validate(batch)
                
                # Update progress bar
                self.progbar.update(values, validating=True)
    
    def check_data_time(self, train_loader):
        """Perform one training loop through the dataloader to check batch data 
        preparation time vs complete batch time."""
        self.progbar = self._make_progbar(train_loader, val_loader=None, verbose=1)
        self.model.train()
        
        t0 = time.time()
        for batch in train_loader:
            t_data = time.time() - t0
                
            _ = self.train(batch)
            
            t_batch = time.time() - t0
            values = {}
            values["t_data"] = t_data
            values["t_batch"] = t_batch
            values["t_data/t_batch"] = t_data / t_batch
                
            # Update progress bar
            self.progbar.update(values)
            
            t0 = time.time()
    
    def save_checkpoint(self, path, metrics_dict=None):
        """Save a checkpoint of the model (the model state_dict and the 
        optimizer state_dict).
        Args:
            path: Path to the file where the checkpoint will be saved.
            metrics_dict: A dictionnary of additional metrics that will be 
                saved.
        """
        # Create directory if necessary
        checkpoint_dir = os.path.join(*path.split("/")[:-1])
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # make checkpoint
        checkpoint = {
        "model_state_dict": self.model.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict()
        }
        if metrics_dict:
            for k in metrics_dict:
                checkpoint[k] = metrics_dict[k]
        
        # save checkpoint
        torch.save(checkpoint, path)
    
    def get_num_parameters(self):
        """Return the total number of parameters of the model."""
        return sum(reduce(lambda a, b: a*b, x.size()) for x in self.model.parameters())
                
    def _make_progbar(self, train_loader, val_loader, verbose):
        """Make a progress bar to show training progress."""
        len_train_loader = len(iter(train_loader))
        if val_loader:
            len_val_loader = len(iter(val_loader))
        else:
            len_val_loader = None
        
        return Progbar(target=len_train_loader, 
                       val_target=len_val_loader, verbose=verbose)