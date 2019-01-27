import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import os
from .utils import Progbar
import time


class Trainer(object):
    """Class abstracting model training in PyTorch."""
    
    def __init__(self, model, loss_fn, optimizer, device="auto"):
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
        
    def get_num_parameters(self):
        """Return the total number of parameters of the model."""
        return sum(reduce(lambda a, b: a*b, x.size()) for x in self.model.parameters())
        
    def fit(self, train_loader, val_loader=None, epochs=1, checkpoint_path=None, 
            early_stopping=False, verbose=1, metrics=None, plot_loss=False):
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
        if metrics:
            assert isinstance(metrics, (tuple, list))
            for metric in metrics:
                assert metric in ("data_time", "batch_time")
        
        if checkpoint_path is not None:
            assert val_loader is not None
            best_val_loss = np.inf
            
        if early_stopping:
            assert train_loader is not None
            assert isinstance(early_stopping, int)
            n = 0
            
        if val_loader:
            val_losses = []
        
        if plot_loss:
            train_losses = []
        
        
        for epoch in range(1, epochs + 1):
            
            print("Epoch {}/{}".format(epoch, epochs))
            self.progbar = self._make_progbar(train_loader, val_loader, verbose)
            
            # Training loop
            self._train_loop(train_loader, metrics)
            
            # Validation loop
            if val_loader:
                self._validate_loop(val_loader, metrics)
            
            # Track losses for plotting
            if plot_loss:
                train_losses.append(self.progbar._values["train_loss"].average())
            if val_loader:
                 val_losses.append(self.progbar._values["val_loss"].average())
            
            # Save best model if improvement on validation loss
            if checkpoint_path and val_losses[-1] < best_val_loss:
                # Save model_dict model_state_dict, optimizer_state_dict, epoch, 
                # and all metrics in progbar.
                metrics_dict = {"Epoch": epoch}
                for k in self.progbar._values.keys():
                    metrics_dict[k] = self.progbar._values[k].average()
                self.save_checkpoint(checkpoint_path, metrics_dict)
                print("Model improved, saved at " + checkpoint_path)
            
            # Check for early stopping
            if early_stopping:
                if val_losses[-1] >= best_val_loss:
                    n += 1
                    if n < early_stopping:
                        print("No improvement in %d Epochs." % n)
                    if n >= early_stopping:
                        print("No improvement in %d Epochs: Early Stopping." % n)
                        break
                else:
                    n = 0
            
            # Update best_val_loss
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                    
        # Plot loss
        if plot_loss:
            plt.figure()
            plt.plot(train_losses, label="Training")
            if val_loader:
                plt.plot(val_losses, label="Validation")
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
        return [("train_loss", loss.item()),
                ("train_acc", self._accuracy(y_pred, y))
                ]
    
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
        return [("val_loss", loss.item()),
                ("val_acc", self._accuracy(y_pred, y))
                ]
    
    def _train_loop(self, train_loader, metrics):
        """Do a single epoch of training."""
        # Set model to train mode
        self.model.train()
        
        t0 = time.time()
        for batch in train_loader:
            values = []
            
            if "data_time" in metrics:
                values.append(("data_time", (time.time() - t0)))
                
            vals = self.train(batch)
            
            if "batch_time" in metrics:
                values.append(("batch_time", (time.time() - t0)))
            
            for val in vals:
                values.append(val)
                
            # Update progress bar
            self.progbar.update(values)
            
            t0 = time.time()
          
            
    def _validate_loop(self, val_loader, metrics):
        """Do a single epoch of validating."""
        # Set model to eval mode
        self.model.eval()
        
        with torch.no_grad():
            for batch in val_loader:
                values = self.validate(batch)
                
                # Update progress bar
                self.progbar.update(values, validating=True)
    
    def save_checkpoint(self, path, metrics_dict=None):
        """Save a checkpoint of the model (the model state_dict and the 
        optimizer state_dict).
        Args:
            path: Path to the file where the checkpoint will be saved.
            metrics_dict: A dictionnary of additional metrics that will be 
                saved.
        """
        # Create directory if necessary
        checkpoint_dir = "".join(path.split("/")[:-1])
        if not os.path.exists(checkpoint_dir) and checkpoint_dir != "":
            os.mkdir(checkpoint_dir)
        
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
                
    def _make_progbar(self, train_loader, val_loader, verbose):
        """Make a progress bar to show training progress."""
        len_train_loader = len(iter(train_loader))
        if val_loader:
            len_val_loader = len(iter(val_loader))
        else:
            len_val_loader = None
        
        return Progbar(target=len_train_loader, 
                       val_target=len_val_loader, verbose=verbose)
        
    def _accuracy(self, y_pred, y):
        """Compute the accuracy for a batch of prediction and target."""
        _, y_pred = torch.max(y_pred, 1)
        return (y_pred == y).cpu().numpy().mean() * 100
