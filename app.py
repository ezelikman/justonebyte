import numpy as np
import sys
import time
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaTokenizer
from transformers import BitsAndBytesConfig
from datasets import load_dataset
from argparse import ArgumentParser
import torch
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
torch.use_deterministic_algorithms(True)
from flask import Flask, request, send_file, Response
from threading import Thread
import hashlib
import pickle
import json
from collections import defaultdict
import bitsandbytes as bnb
import wandb
import datasets
import torch
import math

class Machine:
    def __init__(self, my_address, initial_server_addresses,
                increment_time, buffer_time, inference_time,
                epsilon=0.001, batch_size=16, use_backup=True,
                model_name='gpt2',
                # dataset_name=('sst2', 'main'),
                dataset_name=('glue', 'sst2'),
                dataset_index='question',
                device='best', dtype=torch.float32, use_lora=False, min_num_machines=2, send_full_grad=False,
                normal=False, use_different_gpu=False, debug=False, gradient_acc_steps=1, learning_rate=1e-1, max_iterations = 300,
                use_bnb=False, conditional=True, target_index='label', gamma=1e-1, int_class=False, use_variance_scaling=False,
                one_byte=False,
        ):

        self.my_address = my_address
        self.dataset_name = dataset_name  # Name of the dataset to be used
        self.dataset_index = dataset_index  # Index of the dataset to be used

        self.buffer_time = buffer_time  # Time to stop inferencing before end_time
        self.epsilon = torch.tensor(math.log(epsilon))  # Perturbation size
        self.gamma = gamma # Size of update to epsilon
        if device == 'best':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device  # Device to run model on
        self.model = None  # Model to be trained
        if normal:
            dtype = torch.bfloat16  # Too big otherwise
        self.dtype = dtype  # Data type to use for training
        self.use_lora = use_lora  # Whether to use LoRA
        self.one_byte = one_byte
        self.timestamp = time.time()  # Timestamp used to identify this machine
        self.use_variance_scaling = use_variance_scaling
        self.sample_number = 0  # Number of samples seen so far - reset with each increment
        self.batch_size = batch_size  # Batch size for training
        self.test_batch_size = 64  # Batch size for testing
        self.test_iterations = 64  # Number of iterations to test for
        self.end_time = None  # End of the current increment
        self.increment_time = increment_time  # Length of each increment
        self.projected_grads = []  # Projected gradients for each increment
        self.grad = {}
        self.send_full_grad = send_full_grad
        self.normal=normal
        self.int_class=int_class
        self.inference_time = inference_time  # Time per inference, should be an upper bound
        self.all_addresses = initial_server_addresses  # Addresses of all known servers
        self.min_machine_timestamp = 0.  # Timestamp of the first machine
        self.min_num_machines = min_num_machines  # Minimum number of machines to train with
        if my_address in self.all_addresses:
            self.all_addresses.remove(my_address)
            self.min_machine_timestamp = self.timestamp
        self.addresses_timestamp = {self.my_address: self.timestamp}  # Timestamps of all known servers
        self.dataset = load_dataset(*self.dataset_name)  # Initialize the dataset
        self.train_dataset_indices = np.random.choice(len(self.dataset['train']), 1000, replace=False)  # Indices of the training dataset
        self.validation_dataset_indices = np.random.choice(len(self.dataset['validation']), min(1000, len(self.dataset['validation'])), replace=False)  # Indices of the validation dataset
        self.test_dataset_indices = np.random.choice(len(self.dataset['test']), 16, replace=False)  # Indices of the test dataset
        self.label_names = self.dataset['train'].features['label'].names  # Names of the labels
        self.model_name = model_name  # Name of the model to be used
        if 'llama' in model_name:
            self.postfix = '\nsentiment: '
        else:
            self.postfix = '\nsentiment:\n'
        if "llama" in model_name:
            self.tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/" + model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # Tokenizer for the model
        if self.tokenizer.pad_token is None:  # Add a padding token if there isn't one - common
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if not self.int_class:
            self.class_labels = [
                self.tokenizer.encode(label, add_special_tokens=False) for label in self.label_names
            ]
            # Combine the label tokens into a single one-hot vector where each present label is a 1
            one_hot = np.zeros(len(self.tokenizer))
            for label in self.class_labels:
                one_hot[label] = 1
            self.class_labels = torch.tensor(one_hot).bool()
        else:
            # Treat the string of the integer for each label as a token
            self.class_labels = [self.tokenizer.encode(str(i), add_special_tokens=False) for i in range(len(self.label_names))]
            one_hot = np.zeros(len(self.tokenizer))
            for label in self.class_labels:
                one_hot[label] = 1
            self.class_labels = torch.tensor(one_hot).bool()
        
        self.perturbed = False  # Whether the model has been perturbed (flag to prevent sending weights while perturbed)
        self.sending_weights = False  # Whether the model is currently sending weights (flag to prevent perturbing while sending)
        # We can prevent floating point errors from repeated perturbations by using a backup model
        self.use_backup = use_backup
        self.backup_weights = None
        self.total_iterations = 0
        self.num_finish = defaultdict(set)
        self.calculate_loss = calculate_conditional_loss if conditional else calculate_loss
        self.conditional = conditional
        if self.conditional:
            self.target_index = target_index
        else:
            self.target_index = None
        self.use_different_gpu = use_different_gpu
        self.debug = debug
        self.eval_interval = 128
        self.backup_interval = 16
        self.hash_interval = 64
        assert self.hash_interval % self.backup_interval == 0
        self.gradient_acc_steps=gradient_acc_steps
        self.max_iterations = max_iterations
        self.learning_rate=learning_rate # learning rate for the optimizer; will be overwritten by the main machine if it is not the main machine
        self.app = Flask(__name__)

        self.use_bnb = use_bnb
        if self.use_bnb:
            self.quant_type = "nf4"
            self.unquant_type = torch.bfloat16
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type=self.quant_type,
                bnb_4bit_compute_dtype=self.unquant_type
            )

        # check gpu is available
        assert torch.cuda.is_available(), "No GPU/CUDA is detected!"


        @self.app.route('/model', methods=['GET'])
        def get_model():
            while self.perturbed and (self.backup_weights is None):
                time.sleep(0.1)
            if self.model is not None:
                if self.backup_weights is None:
                    self.sending_weights = True
                    torch.save(self.model.state_dict(), 'temp_model.pt')
                    self.sending_weights = False
                else:
                    torch.save(self.backup_weights, 'temp_model.pt')
                return send_file('temp_model.pt', as_attachment=True)
            else:
                return {'error': 'Model not initialized'}, 500
        
        @self.app.route('/hash', methods=['GET'])
        def get_hash():
            while self.perturbed:
                time.sleep(0.1)
            return {'hash': calculate_hash(self.model, use_bnb=self.use_bnb)}

        @self.app.route('/grads', methods=['GET'])
        def get_grads():
            if self.model is not None:
                if self.send_full_grad or self.normal:
                    if self.normal:
                        self.get_model_grad()
                    return Response(pickle.dumps(self.grad), mimetype='application/octet-stream')
                else:
                    return {'grads': self.projected_grads}
            else:
                return {'error': 'Model not initialized'}, 500
        
        @self.app.route('/notify_finish', methods=['POST'])
        def update_num_finish():
            description = request.json['description']
            self.num_finish[description].add(request.json['address'])
            return {'num_finish': len(self.num_finish[description])}, 200

        @self.app.route('/notify', methods=['POST'])
        def notify_new_server():
            addr, ts = request.json['address'], request.json['timestamp']
            print(f"Received notification from {addr} at {ts}")
            if addr not in self.all_addresses:
                print("New server found at address", addr)
                self.all_addresses.append(addr)
                self.addresses_timestamp[addr] = ts
                for address in self.all_addresses:
                    if address != addr:
                        print(f"Notifying {address} about {addr} at {ts}")
                        try:
                            requests.post(f"{address}/notify", json={'address': addr, 'timestamp': ts})
                        except:
                            print(f"Failed to notify {address} about {addr} at {ts}")
            return {'all_addresses': self.all_addresses, 'timestamps': self.addresses_timestamp, 'end_time': self.end_time, 'learning_rate': self.learning_rate, 'total_iterations': self.total_iterations}, 200

        @self.app.route('/shutdown', methods=['POST'])
        def shutdown():
            self.app.shutdown()
            return 'Server shutting down...'


    ### Initialization functions ###
    def initialize_run(self):
        self.initialize_model()

    def initialize_model(self, use_default=False):
        if use_default or self.total_iterations == 0:
            print("Initializing default model")
            if self.use_bnb:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name if not "llama" in self.model_name else "decapoda-research/" + self.model_name,
                    quantization_config=self.bnb_config,
                    device_map='auto'
                ).eval()
            else:
                if self.normal:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name if not "llama" in self.model_name else "decapoda-research/" + self.model_name,
                        device_map='auto'
                    ).eval()
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name if not "llama" in self.model_name else "decapoda-research/" + self.model_name,
                    ).eval()
            wandb.watch(self.model)
        else:
            random_address = np.random.choice(self.all_addresses)
            print("Getting model from", random_address)
            response = requests.get(f"{random_address}/model")
            print("Received model from", random_address)
            with open('received_model.pt', 'wb') as f:
                f.write(response.content)
            # Load from pretrained config
            config = AutoConfig.from_pretrained(self.model_name if not "llama" in self.model_name else "decapoda-research/" + self.model_name)
            self.model = AutoModelForCausalLM.from_config(config).eval()
        resize_token_embeddings(self.model, len(self.tokenizer))
        if not use_default and self.total_iterations != 0:
            self.model.load_state_dict(torch.load('received_model.pt'))
            print("Loaded model from", random_address)
        if self.model.config.pad_token_id == -1:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = model_processing(self.model, self.dtype, self.device, self.use_lora, self.use_bnb)

    ### Server functions ###
    def start_server(self, port):
        self.app.run(host="0.0.0.0", port=port)
    
    def announce_existence(self):
        for address in self.all_addresses:
            try:
                response = requests.post(f"{address}/notify", json={'address': self.my_address, 'timestamp': self.timestamp}).json()
            except:
                print(f"Could not connect to {address}")
                if self.all_addresses and address not in self.addresses_timestamp:
                    time.sleep(1)
                    self.announce_existence()
                    break
                continue

            for addr in response['all_addresses']:
                if addr not in self.all_addresses and addr != self.my_address:
                    self.all_addresses.append(addr)
            self.addresses_timestamp.update(response['timestamps'])
            self.min_machine_timestamp = min(list(self.addresses_timestamp.values()))
            self.learning_rate = response['learning_rate']
            self.total_iterations = response['total_iterations']

    ### Training functions ###
    def add_perturbation(self, scaling_factor, timestamp, sample_number, use_eps=True, debug=False):
        set_seed(self.total_iterations, timestamp - self.min_machine_timestamp, sample_number)
        if self.gamma > 0:
            eps_z = torch.normal(mean=0, std=1, size=(1,), dtype=self.dtype, device=self.device)
            if use_eps:  # We need to make sure we sample epsilon even if we don't use it, to keep the same random seed
                self.epsilon = self.epsilon + scaling_factor * eps_z * self.gamma
        for param_name, param in self.model.named_parameters():
            if self.use_lora and 'lora' not in param_name:
                continue
            if debug:
                breakpoint()
            if self.use_different_gpu:
                z = torch.normal(mean=0, std=1, size=param.data.size(), dtype=param.data.dtype).to(param.data.device)
            else:
                if self.use_bnb and isinstance(param, bnb.nn.Params4bit):
                    param_dequantized = bnb.functional.dequantize_4bit(param, param.quant_state, quant_type=self.quant_type)
                    z = torch.normal(mean=0, std=1, size=param_dequantized.data.size(), device=param_dequantized.data.device, dtype=param_dequantized.data.dtype)
                    if use_eps:
                        z = z * torch.exp(self.epsilon)
                    if self.use_variance_scaling:
                        scaling_factor = scaling_factor * torch.std(param_dequantized.data) / self.model_variance
                    param_dequantized.data = param_dequantized.data + scaling_factor * z                    
                    param_requantized, param_requantized_state = bnb.functional.quantize_4bit(param_dequantized, quant_type=self.quant_type)
                    param.data = param_requantized.data
                    param.quant_state = param_requantized_state
                else:
                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                    if use_eps:
                        z = z * torch.exp(self.epsilon)
                    if self.use_variance_scaling:
                        if param.data.numel() > 1:
                            scaling_factor = scaling_factor * torch.std(param.data) / self.model_variance
                    param.data = param.data + scaling_factor * z

    def get_model_grad(self):
        for param_name, param in self.model.named_parameters():
            if self.use_lora and 'lora' not in param_name:
                continue
            self.grad[param_name] = param.grad.data.clone().cpu()

    def accumulate_grad(self, scaling_factor, timestamp, sample_number):
        set_seed(self.total_iterations, timestamp - self.min_machine_timestamp, sample_number)
        self.epsilon = self.epsilon + scaling_factor * torch.normal(mean=0, std=1, size=(1,), dtype=self.dtype, device=self.device)
        for param_name, param in self.model.named_parameters():
            if self.use_lora and 'lora' not in param_name:
                continue
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if param_name not in self.grad:
                self.grad[param_name] = scaling_factor * z * torch.exp(self.epsilon)
            else:
                self.grad[param_name] += scaling_factor * z * torch.exp(self.epsilon)

    def apply_full_grad(self, scaling_factor, grad):
        for param_name, param in self.model.named_parameters():
            if self.use_lora and 'lora' not in param_name:
                continue
            param.data = param.data + scaling_factor * grad[param_name].to(param.data.device)
    
    def eval(self):
        self.model.eval()
        losses = []
        start_round_time = time.time()
        with torch.no_grad():
            for i in range(10):
                batch = get_batch(self.batch_size, self.dataset['validation'], self.dataset_index, self.validation_dataset_indices, self.target_index, self.int_class, self.postfix)
                loss = self.calculate_loss(self.model, self.tokenizer, batch, class_labels=self.class_labels)
                losses.append(loss.item())
        model_name = self.model_name.split("/")[-1]
        with open(f'{model_name}_full_grad={self.send_full_grad}_normal={self.normal}_{self.min_machine_timestamp}_{self.learning_rate}.txt', 'a+') as f:
            mean_loss = np.mean(losses)
            f.write(self.my_address+": " + str(mean_loss) + " start eval time: " + str(start_round_time - self.timestamp)  +" finish eval time: " + str(time.time() - self.timestamp) +  " iteration: " + str(self.total_iterations ) + " num samples: 10\n")
            wandb.log({
                "run_id": str(self.min_machine_timestamp),
                "machine_address": self.my_address,
                "mean_eval_loss": mean_loss,
                "start_eval_time": start_round_time - self.timestamp,
                "finish_eval_time": time.time() - self.timestamp,
                "iteration": self.total_iterations,
                "num_eval_samples": 10
            })
        return mean_loss

    def update_weights(self, one_byte=False):
        self.grad = {"num_samples": 0}
        self.projected_grads = []
        self.losses = []
        self.sample_number = 0
        if self.normal:
            self.model.train()
            self.model.zero_grad()

        start_round_time = time.time()
        while self.sample_number < self.gradient_acc_steps and  (time.time() - start_round_time) < self.increment_time:
            init_time = time.time()
            print(f"Sample number: {self.sample_number} - inference time remaining: {self.increment_time + start_round_time - time.time() }")
            while self.sending_weights:
                time.sleep(0.1)
            batch = get_batch(self.batch_size, self.dataset['train'], self.dataset_index, self.train_dataset_indices, self.target_index, self.int_class, self.postfix)
            if self.normal:
                self.grad["num_samples"] += 1
                loss = self.calculate_loss(self.model, self.tokenizer, batch, class_labels=self.class_labels)
                loss.backward()
                loss = loss.detach()
                self.losses.append(loss.item())
                print(f"Projected gradient: disabled  - time elapsed: {time.time() - init_time}, loss = {loss.item()}")
            else:
                self.perturbed = True
                self.add_perturbation(1, self.timestamp, self.sample_number)
                loss_1 = self.calculate_loss(self.model, self.tokenizer, batch, class_labels=self.class_labels).item()
                self.add_perturbation(-2, self.timestamp, self.sample_number)
                loss_2 = self.calculate_loss(self.model, self.tokenizer, batch, class_labels=self.class_labels).item()
                self.add_perturbation(1, self.timestamp, self.sample_number)
                self.perturbed = False
                projected_grad = (loss_1 - loss_2) / (2 * torch.exp(self.epsilon))
                if one_byte:
                    # Make it "just one byte"
                    power = torch.round(torch.log(torch.abs(projected_grad)) * 16)
                    # Clip the power to the range [-127, 127]
                    power = torch.clamp(power, -127, 127)
                    restored_grad = torch.exp(power / 16) * torch.sign(projected_grad)
                self.projected_grads.append(projected_grad.item())
                self.losses.append((loss_1 + loss_2) / 2)
                # if self.use_backup and self.all_addresses:
                if self.use_backup:
                    self.epsilon = self.backup_epsilon
                if self.send_full_grad:
                    self.grad["num_samples"] += 1
                    self.accumulate_grad(self.projected_grads[-1], self.timestamp, self.sample_number)
                print(f"Projected gradient: {self.projected_grads[-1]} - time elapsed: {time.time() - init_time}, loss = {(loss_1 + loss_2) / 2}")
            # If the elapsed time is greater than the inference time, warn the user
            if time.time() - init_time > self.inference_time:
                print("Warning: updating preset inference time to the actual inference time.")
                self.inference_time = time.time() - init_time
            self.sample_number += 1
        if self.sample_number < self.gradient_acc_steps:
            print("Warning: did not have enough samples to reach desired gradients accumulation steps.")
        # Write mean loss to loss.txt
        model_name = self.model_name.split("/")[-1]
        with open(f'{model_name}_full_grad={self.send_full_grad}_normal={self.normal}_{self.min_machine_timestamp}_{self.learning_rate}.txt', 'a+') as f:
            mean_loss = np.mean(self.losses)
            f.write(f'{self.my_address}: train {mean_loss} start inference time: {start_round_time - self.timestamp} finish inference time: {time.time() - self.timestamp} iteration: {self.total_iterations} num samples: {self.sample_number}\n')
            wandb.log({
                "run_id": str(self.min_machine_timestamp),
                "mode": "train",
                "machine_address": self.my_address,
                "mean_train_loss": mean_loss,
                "start_train_time": start_round_time - self.timestamp,
                "finish_train_time": time.time() - self.timestamp,
                "iteration": self.total_iterations,
                "num_train_samples": self.sample_number
            })

    def evaluate_model(self):
        self.sample_number = 0
        self.losses = []
        self.accuracies = []
        self.model.eval()

        start_round_time = time.time()
        # split testing across multiple machines
        n_machines = len(self.all_addresses) + 1
        while self.sample_number < (self.test_iterations // n_machines) and (time.time() - start_round_time) < self.increment_time:
            init_time = time.time()
            print(f"Sample number: {self.sample_number} - inference time remaining: {self.increment_time + start_round_time - time.time() }")
            while self.sending_weights:
                time.sleep(0.1)
            batch = get_batch(self.test_batch_size, self.dataset['validation'], self.dataset_index, self.validation_dataset_indices, self.target_index, self.int_class, self.postfix)

            with torch.no_grad():
                loss, ret_data = self.calculate_loss(self.model, self.tokenizer, batch, return_data=True, class_labels=self.class_labels)
                target_data, logits = ret_data
                predicted_tokens = torch.argmax(logits, dim=-1)[:,:-1]
                target_tokens = target_data['input_ids'][:,1:]
                target_attention = target_data['attention_mask'][:,1:]
                matches = (predicted_tokens == target_tokens) | ~target_attention.bool()
                accuracy = torch.all(matches, dim=-1)
                self.losses.append(loss.item())
                self.accuracies.append(accuracy.float().mean().item())
            print(f"Time elapsed: {time.time() - init_time}, loss = {loss.item()}")

            # If the elapsed time is greater than the inference time, warn the user
            if time.time() - init_time > self.inference_time:
                print("Warning: updating preset inference time to the actual inference time.")
                self.inference_time = time.time() - init_time
            self.sample_number += 1

        if self.sample_number < self.gradient_acc_steps:
            print("Warning: did not have enough samples to reach desired inference steps.")

        # Write mean loss to loss.txt
        model_name = self.model_name.split("/")[-1]
        with open(f'{model_name}_full_grad={self.send_full_grad}_normal={self.normal}_{self.min_machine_timestamp}_{self.learning_rate}.txt', 'a+') as f:
            mean_loss = np.mean(self.losses)
            mean_accuracy = np.mean(self.accuracies)
            f.write(f'{self.my_address}: evaluate {mean_loss} start inference time: {start_round_time - self.timestamp} finish inference time: {time.time() - self.timestamp} iteration: {self.total_iterations} num samples: {self.sample_number}\n')
            wandb.log({
                "run_id": str(self.min_machine_timestamp),
                "mode": "evaluate",
                "machine_address": self.my_address,
                "mean_inference_loss": mean_loss,
                "mean_inference_accuracy": mean_accuracy,
                "start_inference_time": start_round_time - self.timestamp,
                "finish_inference_time": time.time() - self.timestamp,
                "iteration": self.total_iterations,
                "num_inference_samples": self.sample_number
            })

    def request_grads_from_all_machines(self):
        if not self.all_addresses and self.normal:
            self.get_model_grad()
        all_projected_grads = {
            self.my_address: self.projected_grads if not (self.send_full_grad or self.normal) else self.grad
        }
        for address in self.all_addresses:
            try:
                if self.send_full_grad or self.normal:
                    response = requests.get(f"{address}/grads")
                    grad = pickle.loads(response.content)
                else:
                    response = requests.get(f"{address}/grads").json()
                    if not 'grads' in response:
                        print(f"Error: {response['error']}")
                        continue
                    grad = response['grads']
            except Exception as e:
                print(e)
                print(f"Error: could not connect to {address}")
                continue
            all_projected_grads[address] = grad
        return all_projected_grads

    def apply_all_grads(self, all_projected_grads, log=True):
        while self.sending_weights:
            time.sleep(0.1)
        start_round_time = time.time()
        self.perturbed = True
        # Get the sorted list of addresses
        sorted_addresses = sorted(all_projected_grads.keys(), key=lambda x: self.addresses_timestamp[x])
        # Calculate the number of samples
        if self.send_full_grad or self.normal:
            num_samples = sum([all_projected_grads[address]["num_samples"] for address in sorted_addresses])
            for address in sorted_addresses:
                # cur_prop = (self.max_iterations - self.total_iterations) / self.max_iterations
                cur_prop = 1
                self.apply_full_grad(-self.learning_rate / math.sqrt(num_samples) * cur_prop, all_projected_grads[address])
        else:
            num_samples = sum([len(all_projected_grads[address]) for address in sorted_addresses])
            for address in sorted_addresses:
                address_grads = all_projected_grads[address]
                for grad_idx, grad in enumerate(address_grads):
                    if self.debug:
                        breakpoint()
                    # cur_prop = (self.max_iterations - self.total_iterations) / self.max_iterations
                    cur_prop = 1
                    self.add_perturbation(
                        -self.learning_rate * grad / math.sqrt(num_samples) * cur_prop,
                        self.addresses_timestamp[address], grad_idx,
                        use_eps=False, debug=self.debug
                    )
        self.perturbed = False
        model_name = self.model_name.split("/")[-1]
        if log:
            with open(f'{model_name}_full_grad={self.send_full_grad}_normal={self.normal}_{self.min_machine_timestamp}_{self.learning_rate}.txt', 'a+') as f:
                mean_loss = np.mean(self.losses)
                f.write(f'{self.my_address}: apply {mean_loss} start grad time: {start_round_time - self.timestamp} finish grad time: {time.time() - self.timestamp} iteration: {self.total_iterations} num samples: {num_samples}\n')
                wandb.log({
                    "run_id": str(self.min_machine_timestamp),
                    "machine_address": self.my_address,
                    "start_grad_time": start_round_time - self.timestamp,
                    "finish_grad_time": time.time() - self.timestamp,
                    "iteration": self.total_iterations,
                    "num_grad_samples": num_samples
                })


    def notify_finish(self, description="initialize"):
        for address in self.all_addresses:
            try:
                response = requests.post(f"{address}/notify_finish" , json={'address': self.my_address, 'description': description})
            except Exception as e:
                print(e)
                print(f"Error: could not connect to {address}")
                continue

    def wait_till_finish(self, count=None, description="initialize", timeout=100, quit_on_timeout=True):
        # wait for everyone to finish
        start_time = time.time()
        if count is None:
            count = len(self.all_addresses)
        printed_waiting = False
        while (len(self.num_finish[description]) < count) and (time.time() - start_time < timeout): # exclude the current machine
            if not printed_waiting:
                print(f"Waiting for machines to {description}... currently {len(self.num_finish[description]) + 1}/{count + 1}")
                printed_waiting = True
            time.sleep(0.1)
        if len(self.num_finish[description]) < count:
            if quit_on_timeout:
                print(f"Timed out waiting for machines to finish {description}.")
                sys.exit()
            missing_addresses = set(self.all_addresses) - self.num_finish[description]
            print(f"Warning: {missing_addresses} did not {description} in time, removing them from the list of machines.")
            self.all_addresses = list(set(self.all_addresses) - self.num_finish[description])
        print(f"All machines finished {description}.")
        self.num_finish[description] = set()

    def sync(self, description="initialize", count=None):
        if count is None:
            count = len(self.all_addresses)
        self.notify_finish(description)
        self.wait_till_finish(count, description, timeout = 200000)

    def calculate_variance(self):
        self.model_variance = 0
        self.total_param_items = 0
        for param in self.model.parameters():
            self.model_variance += torch.var(param.data)
            self.total_param_items += 1
        self.model_variance /= self.total_param_items
        # We actually want the standard deviation
        self.model_variance = torch.sqrt(self.model_variance)

    def restore_grads(self, backup_grads):
        true_iteration = self.total_iterations
        for past_iteration, past_grads in backup_grads:
            self.total_iterations = past_iteration
            self.apply_all_grads(past_grads, log=False)
        self.total_iterations = true_iteration

    def run(self):
        num_joined = 0
        self.announce_existence()
        if not self.all_addresses:
            print("No other machines found.")
            self.initialize_run()
        if self.model is None:
            print("Model not initialized.")
            self.initialize_model()
        self.sync("finish initialize model", max(self.min_num_machines-1, len(self.all_addresses)))
        while self.total_iterations < self.max_iterations:  # Run the training loop
            with torch.inference_mode(mode = not self.normal):
                print("Starting run.")
                self.announce_existence()
                if self.model is None:
                    print("Model not initialized.")
                    self.initialize_model()
                # if self.use_backup and self.all_addresses: 
                if self.use_backup and self.total_iterations % self.backup_interval == 0:
                    print("Backing up weights.")
                    print(f"start backup {time.time()}")
                    self.backup_weights = {k: v.cpu() for k, v in self.model.state_dict().items()}
                    self.backup_epsilon = self.epsilon
                    self.backup_grads = []
                    print(f"end backup {time.time()}")
                    print(f"Setting epsilon to {self.backup_epsilon}")
                if self.use_variance_scaling:
                    # To use variance scaling, we calculate the overall variance of the model
                    # This isn't actually necessary, but it allows us to use the same epsilon
                    # Across different parts of the model
                    print("Calculating variance.")
                    self.calculate_variance()
                print("Calculating losses.")
                self.update_weights(one_byte=self.one_byte)
                self.sync("finish forward pass")
                if self.use_backup and (self.total_iterations + 1) % self.backup_interval == 0:
                    print("Restoring weights.")
                    self.model.load_state_dict(self.backup_weights)
                    if len(self.backup_grads) > 0:
                        print("Restoring gradients.")
                        restore_grads_time = time.time()
                        self.restore_grads(self.backup_grads)
                        print(f"Restoring gradients took {time.time() - restore_grads_time} seconds.")
                    self.backup_grads = []
                    self.sync("finish restoring weights")
                print("Requesting gradients.")
                all_projected_grads = self.request_grads_from_all_machines()
                # self.backup_grads[self.total_iterations] = all_projected_grads
                self.backup_grads.append((self.total_iterations, all_projected_grads))
                print("Applying gradients.")
                self.apply_all_grads(all_projected_grads)
                self.sync("finish applying gradients")
                if self.all_addresses and (self.total_iterations + 1) % self.hash_interval == 0:
                    print(f"start hashing {time.time()}")
                    self.model = confirm_hash(np.random.choice(self.all_addresses), self.model)
                    print(f"finish hashing {time.time()}")
                self.total_iterations += 1
                full_batch_size = (len(self.all_addresses) + 1) * self.gradient_acc_steps
                if (self.total_iterations - 1) % (1 + (self.eval_interval // full_batch_size)) == 0:
                    print("Evaluating model.")
                    self.evaluate_model()
                self.sync("finish round")
                print(f"Finished training for iteration {self.total_iterations} ending at", time.time())

        self.sync("exit")

def calculate_conditional_loss(model, tokenizer, batch_tuple, return_data=False, class_labels=None):
    batch_in, batch_out = batch_tuple
    # Concatenate the strings
    batch = [in_str + out_str for in_str, out_str in zip(batch_in, batch_out)]
    tokenized_batch = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, return_token_type_ids=False if "llama" in tokenizer.name_or_path else None)
    device = next(model.parameters()).device
    tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}
    
    outputs = model(**tokenized_batch)

    labels = tokenized_batch["input_ids"].clone()
    logits = outputs.logits
    # Ignore batch_in tokens
    for i, in_str in enumerate(batch_in):
        if in_str[-1] == ' ':
            in_str = in_str[:-1]
        labels[i, :len(tokenizer.encode(in_str))] = -100
        tokenized_batch["attention_mask"][i, :len(tokenizer.encode(in_str))] = 0
    # Ignore padding tokens
    labels[labels == tokenizer.pad_token_id] = -100
    logits[:, :, tokenizer.pad_token_id] -= 100
    if class_labels is not None:
        # Ignore tokens that are not in the class labels
        logits[:, :, ~class_labels] -= np.inf
    # Get rid of the last token of logits and the first token of labels
    train_logits = logits[:, :-1]
    # Filter to only the tokens that are in the labels
    target_labels = labels[:, 1:]
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(
        train_logits.reshape(
            -1, train_logits.size(-1)), target_labels.reshape(-1))
    del outputs
    if return_data:
        # Remove the input_ids from attention_mask
        return loss, (tokenized_batch, logits)
    return loss

def calculate_loss(model, tokenizer, batch, return_data=False, class_labels=None):
    tokenized_batch = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, return_token_type_ids=False if "llama" in tokenizer.name_or_path else None)
    if class_labels is not None:
        print("Warning: class_labels is not None, but it is not used in calculate_loss.")
    device = next(model.parameters()).device
    tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}
    outputs = model(**tokenized_batch, labels=tokenized_batch["input_ids"])
    loss = outputs.loss
    if return_data:
        return loss, (tokenized_batch, outputs.logits)
    del outputs
    return loss

def get_batch(batch_size, dataset, dataset_index, indices, dataset_target_index=None, int_class=False, postfix="\nsentiment: "):
    # Randomly choose indices for batch sampling
    batch_size = min(batch_size, len(indices))
    indices = list(np.random.choice(indices, size=batch_size, replace=False))
    batch = dataset[indices]  # Select the batch from the dataset
    if dataset_target_index is None:
        batch = batch[dataset_index]
        batch = [text for text in batch if text.strip()]  # Filter empty strings
    else:
        batch_in = batch[dataset_index]
        target = batch[dataset_target_index]
        target_feature = dataset.features[dataset_target_index]
        if isinstance(target_feature, datasets.ClassLabel) and not int_class:
            target_map = target_feature.int2str
        else:
            target_map = lambda x: str(x)
        batch = list(zip(*[(str(text + postfix), target_map(target)) for text, target in zip(batch_in, target) if text.strip()]))
    if len(batch) < 3 and batch_size >= 3 and dataset_target_index is None:  # If the batch size doesn't match, try again
        return get_batch(batch_size, dataset, dataset_index, indices, dataset_target_index, int_class, postfix)
    return batch

def model_processing(model, dtype, device, use_lora, use_bnb):
    if not use_bnb:
        model.to(dtype)  # Process a model after loading it
        model.eval()
        model.to(device)
    if use_lora:
        from peft import get_peft_model, LoraConfig, TaskType
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=True, r=8, lora_alpha=32, lora_dropout=0.0
        )
        model = get_peft_model(model, peft_config)
    return model

def set_seed(total_iterations, timestamp, sample_number):
    timer = int(f'{timestamp:.6f}'.replace('.', ''))
    full_seed = int(f'{total_iterations}{sample_number:05}{timer}') % 9223372036854775806
    torch.manual_seed(full_seed)
    torch.cuda.manual_seed(full_seed)

def resize_token_embeddings(model, new_size):
    cur_seed = torch.seed()
    torch.manual_seed(0)  # For reproducibility
    model.resize_token_embeddings(new_size)
    torch.manual_seed(cur_seed)

def calculate_hash(model, decimals=3, use_bnb=False):
    # Calculates the hash of the model to verify the model is the same on all machines
    # Ignore differences on the order of epsilon, so we round
    str_model = ""
    for _, param in model.named_parameters():
        if use_bnb and isinstance(param, bnb.nn.Params4bit):
            param_dequantized = bnb.functional.dequantize_4bit(param, param.quant_state, quant_type=self.quant_type)
            str_model += str((param_dequantized.data * 10 ** decimals).round().int().cpu().numpy())
        else:
            str_model += str((param.data * 10 ** decimals).round().int().cpu().numpy())
    serialized_model = str_model.encode()
    model_hash = hashlib.md5(serialized_model).hexdigest()
    print(f"Model hash: {model_hash}")
    return model_hash

def confirm_hash(address, model, use_bnb=False):
    try:
        response = requests.get(f"{address}/hash")
    except:  # If the hash don't match, the model is reset
        print("Error: No response from address, assuming hash is correct.")
    if calculate_hash(model, use_bnb=use_bnb) != response.json()['hash']:
        import pdb; pdb.set_trace()
        print("Hashes don't match.")
        model = None
    else:
        print("Hashes match.")
    return model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--port', type=int, default=7000)
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--dataset_name', type=tuple, default=('sst2', 'main'))
    parser.add_argument('--dataset_index', type=str, default="sentence")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epsilon', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.)
    parser.add_argument('--increment_time', type=float, default=10000)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--max_iterations', type=int, default=4000)
    parser.add_argument('--gradient_acc_steps', type=float, default=30)
    parser.add_argument('--buffer_time', type=float, default=0)
    parser.add_argument('--use_variance_scaling', type=bool, default=False)
    parser.add_argument('--inference_time', type=float, default=1)
    parser.add_argument('--min_num_machines', type=int, default=2)
    parser.add_argument('--send_full_grad', type=bool, default=False)
    parser.add_argument('--normal', type=bool, default=False)
    parser.add_argument('--use_different_gpu', type=bool, default=False)
    parser.add_argument('--use_bnb', type=bool, default=False)
    parser.add_argument('--int_class', type=bool, default=False)
    parser.add_argument('--start_ip', type=str, default= "127.0.0.1")
    parser.add_argument('--self_ip', type=str, default= "127.0.0.1")
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--one_byte', type=bool, default=False)
    parser.add_argument('--conditional', type=bool, default=True)
    parser.add_argument('--start_port', type=int, default=7000)
    
    args = parser.parse_args()
    max_iterations = args.max_iterations / args.gradient_acc_steps
    server = Machine(f'http://{args.self_ip}:{args.port}', [f'http://{args.start_ip}:{args.start_port}'], args.increment_time, args.buffer_time, args.inference_time, epsilon=args.epsilon, batch_size=args.batch_size, model_name=args.model_name, min_num_machines=args.min_num_machines, send_full_grad=args.send_full_grad, normal=args.normal, use_different_gpu=args.use_different_gpu, debug=args.debug, gradient_acc_steps=args.gradient_acc_steps, learning_rate=args.learning_rate, max_iterations=args.max_iterations, dataset_name=args.dataset_name, dataset_index=args.dataset_index, use_bnb=args.use_bnb, conditional=args.conditional, gamma=args.gamma, int_class=args.int_class, use_variance_scaling=args.use_variance_scaling, one_byte=args.one_byte)
    wandb.init(project="justonebyte", name = f'{args.model_name}_full_grad={args.send_full_grad}_normal={args.normal}_{server.timestamp}_{args.self_ip}_{args.learning_rate}_{args.epsilon}_{args.gamma}')

    #save config
    wandb.config.update(args)

    t = Thread(target=server.start_server, args=(args.port,))
    t.daemon = True
    t.start()
    try:
        server.run()
    except Exception as e:
        import traceback
        traceback.print_exc()
        pass
    sys.exit()
