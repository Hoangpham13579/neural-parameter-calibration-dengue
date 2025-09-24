import sys
from os.path import dirname as up

import h5py as h5
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchdiffeq import odeint
from dantro import logging
from dantro._import_tools import import_module_from_path

sys.path.append(up(up(__file__)))
sys.path.append(up(up(up(__file__))))

Dengue = import_module_from_path(mod_path=up(up(__file__)), mod_str="Dengue")
base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------
# Model implementation
# ----------------------------------------------------------------------------------------------------------------------


class Dengue_CNN_LSTM:
    def __init__(
        self,
        *,
        rng: np.random.Generator,
        h5group: h5.Group,
        cnn_lstm: base.CNN_LSTM,
        loss_function: dict,
        to_learn: list,
        true_parameters: dict = {},
        dt: float,
        Dengue_data_loss: bool = False,
        write_every: int = 1,
        write_start: int = 1,
        training_data: torch.Tensor,
        batch_size: int,
        num_epochs: int = 100,
        grad_clip: float = None,
        device: str = None,
        parameter_ranges: list = None,
        **__,
    ):
        """Initialize the model instance with a previously constructed RNG and
        HDF5 group to write the output data to.

        Args:
            rng (np.random.Generator): The shared RNG
            h5group (h5.Group): The output file group to write data to
            cnn_lstm: The CNN-LSTM model
            loss_function (dict): the loss function to use
            to_learn: the list of parameter names to learn
            time_dependent_parameters: dictionary of time-dependent parameters and their granularity
            true_parameters: the dictionary of true parameters
            dt: time differential
            Dengue_data_loss: whether to use the loss structure unique to the Dengue data
            write_every: write every iteration
            write_start: iteration at which to start writing
            training_data: the training data to use
            batch_size: epoch batch size: instead of calculating the entire time series,
                only a subsample of length batch_size can be used. The likelihood is then
                scaled up accordingly.
        """
        self._h5group = h5group
        self._rng = rng

        self.cnn_lstm = cnn_lstm
        # Optimizer
        self.cnn_lstm.optimizer.zero_grad()
        self.loss_function = base.LOSS_FUNCTIONS[loss_function.get("name").lower()](
            loss_function.get("args", None), **loss_function.get("kwargs", {})
        )

        self.dt = torch.tensor(dt, dtype=torch.float)
        self.Dengue_data_loss = Dengue_data_loss

        self.current_loss = torch.tensor(0.0)

        self.to_learn = {key: idx for idx, key in enumerate(to_learn)}

        self.true_parameters = {
            key: torch.tensor(val, dtype=torch.float)
            for key, val in true_parameters.items()
        }
        self.all_parameters = set(self.to_learn.keys())
        self.all_parameters.update(self.true_parameters.keys())
        self.current_predictions = torch.zeros(len(self.to_learn), dtype=torch.float)

        # Preprocess data for CNN-LSTM model
        self.training_data = training_data  # (batch_size, #_latent_var, 1)
        
        # Generate the batch ids (# of batches from the training dataset)
        self.batches = np.arange(0, self.training_data.shape[0]+1, batch_size)
        if self.batches[-1] < self.training_data.shape[0]:
            self.batches = np.append(self.batches, training_data.shape[0] - 1)

        self.batch_size = batch_size
        self.grad_clip = grad_clip  # Clip gradient within [-grad_clip, grad_clip]
        self.num_epochs = num_epochs
        self.device = device

        # LR scheduler
        if num_epochs > 0:
            self.lr_scheduler = optim.lr_scheduler.OneCycleLR(
                self.cnn_lstm.optimizer,
                max_lr=self.cnn_lstm.learning_rate,
                div_factor=10,  # Determines initial lr via initial_lr = max_lr/div_factor
                pct_start=0.2,  # The percentage of the cycles spent increasing the learning rate
                epochs=num_epochs,
                steps_per_epoch=len(self.batches),
                anneal_strategy='cos',  # Cosine annealing
            )

        # # Reduce the lr automatically when the loss stops decreasing
        # self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.cnn_lstm.optimizer,
        #     mode='min',
        #     factor=0.5,
        #     patience=5,
        #     verbose=True,
        # )

        # # Cosine Annealing with Warm Restarts
        # self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     self.cnn_lstm.optimizer,
        #     T_0=20,  # Initial restart period
        #     T_mult=2,  # Multiplier for increasing restart period
        #     eta_min=0.0001,  # Minimum learning rate
        # )
        
        # --- Set up chunked dataset to store the state data in --------------------------------------------------------
        self._dset_loss = self._h5group.create_dataset(
            "loss",
            (0,),
            maxshape=(None,),
            chunks=True,
            compression=3,
        )
        self._dset_loss.attrs["dim_names"] = ["batch"]
        self._dset_loss.attrs["coords_mode__batch"] = "start_and_step"
        self._dset_loss.attrs["coords__batch"] = [write_start, write_every]

        self.dset_time = self._h5group.create_dataset(
            "computation_time",
            (0,),
            maxshape=(None,),
            chunks=True,
            compression=3,
        )
        self.dset_time.attrs["dim_names"] = ["epoch"]
        self.dset_time.attrs["coords_mode__epoch"] = "trivial"

        # Create a dataset for the parameter estimates
        self.dset_parameters = self._h5group.create_dataset(
            "parameters",
            (0, len(self.to_learn.keys())),
            maxshape=(None, len(self.to_learn.keys())),
            chunks=True,
            compression=3,
        )
        self.dset_parameters.attrs["dim_names"] = ["batch", "parameter"]
        self.dset_parameters.attrs["coords_mode__batch"] = "start_and_step"
        self.dset_parameters.attrs["coords__batch"] = [write_start, write_every]
        self.dset_parameters.attrs["coords_mode__parameter"] = "values"
        self.dset_parameters.attrs["coords__parameter"] = to_learn

        # --------------------------------------------------------------------------------------------------------------
        # Batches processed
        self._time = 0
        self._write_every = write_every
        self._write_start = write_start
        
        # Hi0, Me0, Hr0, Hs0, He0, Ms0, Mi0, beta_h, beta_m, gamma_h, lambda, mu_m, theta_h, theta_m
        self.parameter_ranges = parameter_ranges[1:]

    def apolo_model_7(self, t, y, params):
        """ODE system for the Dengue model
        Args:
            t: Time point
            y: State vector [Ms, Me, Mi, Hs, He, Hi, Hr, Itot]
            parameters: Dictionary containing model parameters
        """
        # Ms, Me, Mi, Hs, He, Hi, Hr, Itot = y
        Itot, Hi, Me, Hr, Hs, He, Ms, Mi = y
        lambda_, beta_m, mu_m, theta_m, mu_h, beta_h, theta_h, gamma_h = params
    
        # Calculate total populations
        H = Hr + Hs + He + Hi  # Total human population
        M = Me + Ms + Mi       # Total mosquito population
    
        # Ensure populations are not zero
        H = torch.clamp(H, min=1e-8)
        M = torch.clamp(M, min=1e-8)
    
        # ODE system
        dydt = torch.stack([
            theta_h * He,
            theta_h * He - (gamma_h + mu_h) * Hi,
            beta_m * Hi * Ms / H - (theta_m + mu_m) * Me,
            gamma_h * Hi - mu_h * Hr,
            mu_h * H - beta_h * Mi * Hs / M - mu_h * Hs,
            beta_h * Mi * Hs / M - (theta_h + mu_h) * He,
            lambda_ - beta_m * Hi * Ms / H - mu_m * Ms,
            theta_m * Me - mu_m * Mi,
        ])
        
        return dydt
    
    def rescale_parameters(self, predicted_parameters, ranges):
        """
        Scale predicted parameters tensor to their specific ranges using sigmoid normalization.
        
        Args:
            predicted_parameters (torch.Tensor): Tensor of predicted parameters (#_to_learn,)
            ranges (list): List of (min, max) tuples for each parameter
            
        Returns:
            torch.Tensor: Scaled parameters within their specified ranges
        """
        # Scale each parameter to its range
        scaled_parameters = torch.zeros_like(predicted_parameters)
        for i, (min_val, max_val) in enumerate(ranges):
            scaled_parameters[i] = predicted_parameters[i] * (max_val - min_val) + min_val

        return scaled_parameters
    
    def epoch(self):
        """Trains the model for a single epoch"""

        # Go through each batch
        # self.batches: [0, batch_size, 2*batch_size, ..., len(training_data)]
        for batch_no, batch_idx in enumerate(self.batches[:-1]):
            # (NOTE) Input data point at t=0 of Itot compartment only to predict the future outbreak (not the whole dataset)
            batch_data = self.training_data[batch_idx, 0, :].unsqueeze(dim=0)  #(1, 1)

            # (NOTE) Each set of parameters is used to estimate the whole batch latent variables --> predicted_parameters has shape (#_latent_var,) only
            predicted_parameters = self.cnn_lstm(batch_data)  # (#_to_learn,)
            
            # Ensure the predicted parameters are non-zero
            temp = 1
            while (torch.any(predicted_parameters == 0) or torch.isnan(predicted_parameters).any()):
                log.info(f"Predicted parameters contain zero-value: {predicted_parameters}")
                batch_data = self.training_data[batch_idx + temp, 0, :]  #(1, 1)
                predicted_parameters = self.cnn_lstm(batch_data)  # (#_to_learn)
                temp += 1
                
            # Scale the parameters using the new function
            predicted_parameters = self.rescale_parameters(predicted_parameters, self.parameter_ranges)
            
            # Store predicted parameters in a dictionary
            parameters = {
                p: (
                    predicted_parameters[self.to_learn[p]]
                    if p in self.to_learn.keys()
                    else self.true_parameters[p]
                )  # (batch_size,)
                for p in self.all_parameters
            }
            
            Itot0 = torch.tensor(float(batch_data[0].squeeze()), dtype=torch.float32).to(self.device)
            y0 = torch.stack([Itot0, parameters["Hi0"], parameters["Me0"], 
                               parameters["Hr0"], parameters["Hs0"], 
                               parameters["He0"], parameters["Ms0"], 
                               parameters["Mi0"]])
            params = torch.stack([parameters['lambda'], parameters['beta_m'], 
                                  parameters['mu_m'], parameters['theta_m'], 
                                  parameters['mu_h'].to(self.device), parameters['beta_h'],
                                  parameters['theta_h'], parameters['gamma_h']
                                  ])
            
            
            # Setup time points for integration
            t_true = torch.arange(0, self.batches[batch_no + 1] + 1, dtype=torch.float32)

            # Solve ODE system
            solution = odeint(func=lambda y, t: self.apolo_model_7(y, t, params), 
                              y0=y0, t=t_true,
                              method='dopri5',     # dopri5, euler, or rk4
                              rtol=1e-3,          # Relaxed tolerance
                              atol=1e-6,          # Relaxed absolute tolerance
            )
            while torch.isnan(solution).any() or torch.isinf(solution).any():
                solution = odeint(func=lambda y, t: self.apolo_model_7(y, t, params), 
                                  y0=y0, t=t_true,
                                  method='euler',     # dopri5, euler, or rk4
                                  rtol=1e-3,          # Relaxed tolerance
                                  atol=1e-6,          # Relaxed absolute tolerance
                )
            # Ensure non-negative values & discard initial condition
            pred_densities = torch.abs(solution[1:]).to(self.device)  # (batch_size, #_latent_var)
            obser_densities = self.training_data[
                self.batches[batch_no] : self.batches[batch_no + 1], :, :
            ].squeeze().to(self.device)  # (batch_size, #_latent_var, 1)
            
            # Apply min-max scaling to both predictions and observations
            def min_max_scale(tensor):
                min_vals = tensor.min(dim=0, keepdim=True)[0]
                max_vals = tensor.max(dim=0, keepdim=True)[0]
                scaled = (tensor - min_vals) / (max_vals - min_vals + 1e-8)
                return scaled
        
            # Scale predictions and observations
            pred_densities_scaled = min_max_scale(pred_densities)
            obser_densities_scaled = min_max_scale(obser_densities)
            
            # Compute loss
            total_loss = self.loss_function(pred_densities_scaled[:, 0],
                                            obser_densities_scaled[:, 0]).sum()
                
            # Perform a gradient descent step
            total_loss.backward()
            log.info(f"Gradient norm: {str(self.total_gradient_norm())}")
            
            # Gradient Clipping
            if self.grad_clip:
                torch.nn.utils.clip_grad_value_(
                    self.cnn_lstm.parameters(), self.grad_clip
                )

            self.cnn_lstm.optimizer.step()
            self.lr_scheduler.step() if self.num_epochs > 0 else None
            self.cnn_lstm.optimizer.zero_grad()
            self.current_loss = total_loss.clone().detach().cpu().numpy().item()
            self.current_predictions = final_parameters.clone().detach().cpu()  # (#_estimated_params,)
            self._time += 1
            self.write_data()
            
    def total_gradient_norm(self):
        """Calculate the total norm of gradients across all parameters"""
        total_norm = 0.0
        for p in self.cnn_lstm.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm**0.5
    
    # Min-max normalization while preserving gradients
    def min_max_normalize(self, X, feature_range=(0, 1)):
        # Min-max values
        min_val = X.min(axis=0).values  # Get only the values, not indices
        max_val = X.max(axis=0).values
        
        # Normalize
        X_std = (X - min_val) / (max_val - min_val + 1e-8)
        
        # Scale to feature range
        X_scaled = X_std * (feature_range[1] - feature_range[0]) + feature_range[0]
        return X_scaled

    def write_data(self):
        """Write the current state (loss and parameter predictions) into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        """
        if self._time >= self._write_start and (self._time % self._write_every == 0):
            # Log the loss values (along epoch)
            self._dset_loss.resize(self._dset_loss.shape[0] + 1, axis=0)
            self._dset_loss[-1] = self.current_loss

            # Log predicted latent variables into .h5 file (along data-point)
            self.dset_parameters.resize(self.dset_parameters.shape[0] + 1, axis=0)
            self.dset_parameters[-1, :] = [
                self.current_predictions[self.to_learn[p]] for p in self.to_learn.keys()
            ]
