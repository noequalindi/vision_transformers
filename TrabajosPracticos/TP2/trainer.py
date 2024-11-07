"""
trainer.py

Author: Abraham Rodriguez
DATE: 24/5/2023
"""
import copy
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import copy

class EarlyStopping():
    def __init__(self, patience: int = 5, min_delta: float = 0, restore_best_weights: bool = True, verbose: bool = False):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose  
        self.best_model = None
        self.best_loss = float('inf')
        self.counter = 0
        self.status = ""

    def __call__(self, model: torch.nn.Module, val_loss: float):
        if self.best_loss == float('inf'):
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
            if self.verbose:
                print(f"Epoch 0: Inicialización - Best loss: {self.best_loss}")

        elif self.best_loss - val_loss > self.min_delta:
            # Si la pérdida mejora (es decir, si la nueva pérdida es más baja)
            self.best_loss = val_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())  # Guardar el mejor modelo
            if self.verbose:
                print(f"Mejora en la pérdida: {val_loss} (Best loss actualizada a {self.best_loss})")

        elif self.best_loss - val_loss < self.min_delta:
            # Si no hay mejora significativa
            self.counter += 1
            if self.verbose:
                print(f"Epoch {self.counter}/{self.patience}: No mejora significativa, val_loss: {val_loss}")
            
            if self.counter >= self.patience:
                # Si se alcanza el límite de paciencia, detener el entrenamiento
                self.status = f"Entrenamiento detenido en la época {self.counter}"
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model.state_dict())  # Restaurar el mejor modelo
                if self.verbose:
                    print(f"Parada temprana activada después de {self.counter} épocas sin mejora.")
                return True

        self.status = f"{self.counter}/{self.patience}"
        return False

class Trainer():
	"""
	Custom trainer Class that wraps the training and evaluation of a model, using torch autocast
	"""
	def __init__(self, model : torch.nn.Module, train_data_loader: DataLoader,
							test_data_loader: DataLoader ,loss_fn:torch.nn.Module,
								optimizer: torch.optim.Optimizer, device: str):
		"""

		Class constructor, sets mechanism to a certain quantity of patience, and a defined min_delta,
		and the best weights of the trained model.

		:param model : patience to stop
		:type model : torch.nn.Module

		:param train_data_loader : minimum difference between losses per epoch.
		:type train_data_loader : torch.utils.data.DataLoader

		:param test_data_loader :  restore best model
		:type test_data_loader : torch.utils.data.DataLoader

		:param loss_fn :  restore best model
		:type loss_fn : torch.nn.Module

		:param optimizer :  restore best model
		:type optimizer: torch.optim.Optimizer

		:param device :  restore best model
		:type device: str

		"""
		self.model = model
		self.train_data_loader = train_data_loader
		self.test_data_loader = test_data_loader
		self.loss_fn = loss_fn
		self.metrics = None
		self.optimizer = optimizer
		self.device = device

	# @property
	# def device(self):
	# 	return self.device

	# @device.setter
	# def device(self, new_device : str ):
	# 	self.device = new_device

	# @device.deleter
	# def device(self):
	# 	del self.device

	def train_model(self,use_amp = False, dtype : torch.dtype = torch.bfloat16):

		model = self.model.train()
		#scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
		losses = []
		bar = tqdm(self.train_data_loader)
		for train_input, train_mask in bar:
				self.optimizer.zero_grad()
				train_mask = train_mask.to(self.device)
				train_input=train_input.to(self.device)
				with torch.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
					output = model(train_input)
					loss = self.loss_fn(output, train_mask)
				# if isinstance(dtype, type(torch.float16)):
				# 	scaler.scale(loss).backward()
				# 	scaler.step(self.optimizer)
				# 	scaler.update()
				# else:
					
				loss.backward()
				self.optimizer.step()

				# outputs=model(train_input.float())
				# loss = loss_fn(outputs.float(), train_mask.float())
				losses.append(loss.item())
				#loss.backward()
				#optimizer.step()
				#optimizer.zero_grad()
				for param in model.parameters():
					param.grad = None
				bar.set_description(f"loss {loss:.5f}")
		return np.mean(losses)
      
	def eval_model(self):
		model = self.model.eval()

		losses = []
		correct_predictions = 0
		total_predictions = 0
		bar = tqdm(self.test_data_loader)

		with torch.no_grad():
			for val_input, val_mask in bar:
				val_mask = val_mask.to(self.device)
				val_input = val_input.to(self.device)
				outputs = model(val_input)

				# Calcular la pérdida
				loss = self.loss_fn(outputs, val_mask)
				losses.append(loss.item())

				# Calcular la precisión
				_, preds = torch.max(outputs, 1)
				correct_predictions += (preds == val_mask).sum().item()
				total_predictions += val_mask.size(0)

				bar.set_description(f"val_loss {loss:.5f}")

		# Calcular la precisión como porcentaje
		accuracy = correct_predictions / total_predictions * 100
		return np.mean(losses), accuracy