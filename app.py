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
# torch.cuda.memory._record_memory_history(True)


# def oom_observer(device, alloc, device_alloc, device_free):
#     # snapshot right after an OOM happened
#     print('saving allocated state during OOM')
#     snapshot = torch.cuda.memory._snapshot()
#     pickle.dump(snapshot, open('oom_snapshot.pickle', 'wb'))

# torch._C._cuda_attach_out_of_memory_observer(oom_observer)

class Machine:
    def __init__(self, my_address, initial_server_addresses,
                increment_time, buffer_time, inference_time,
                epsilon=0.001, batch_size=16, use_backup=True,
                model_name='gpt2', dataset_name=('sst2', 'main'), dataset_index='question',
                device='best', dtype=torch.float16, use_lora=False, min_num_machines=2, send_full_grad=False,
                normal=False, use_different_gpu=False, debug=False, gradient_acc_steps=1, learning_rate=1e-1, max_iterations = 300,
                use_bnb=False, conditional=True, target_index='label'
        ):
        self.my_address = my_address
        self.dataset_name = dataset_name  # Name of the dataset to be used
        self.dataset_index = dataset_index  # Index of the dataset to be used
        self.buffer_time = buffer_time  # Time to stop inferencing before end_time
        self.epsilon = epsilon  # Perturbation size
        if device == 'best':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device  # Device to run model on
        self.model = None  # Model to be trained
        self.dtype = dtype  # Data type to use for training
        self.use_lora = use_lora  # Whether to use LoRA
        self.timestamp = time.time()  # Timestamp used to identify this machine

        self.sample_number = 0  # Number of samples seen so far - reset with each increment
        self.batch_size = batch_size  # Batch size for training
        self.end_time = None  # End of the current increment
        self.increment_time = increment_time  # Length of each increment
        self.projected_grads = []  # Projected gradients for each increment
        self.grad = {}
        self.send_full_grad = send_full_grad
        self.normal=normal
        self.inference_time = inference_time  # Time per inference, should be an upper bound
        self.all_addresses = initial_server_addresses  # Addresses of all known servers
        self.min_machine_timestamp = 0.  # Timestamp of the first machine
        self.min_num_machines = min_num_machines  # Minimum number of machines to train with
        if my_address in self.all_addresses:
            self.all_addresses.remove(my_address)
        self.addresses_timestamp = {self.my_address: self.timestamp}  # Timestamps of all known servers
        self.dataset = load_dataset(*self.dataset_name)  # Initialize the dataset
        self.model_name = model_name  # Name of the model to be used
        if "llama" in model_name:
            self.tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/" + model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # Tokenizer for the model
        if self.tokenizer.pad_token is None:  # Add a padding token if there isn't one - common
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
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
        self.eval_interval = 1
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
            return {'hash': calculate_hash(self.model)}

        @self.app.route('/grads', methods=['GET'])
        def get_grads():
            if self.model is not None:
                if self.send_full_grad or self.normal :
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
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name if not "llama" in self.model_name else "decapoda-research/" + self.model_name
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
    def add_perturbation(self, scaling_factor, timestamp, sample_number, debug=False):
        set_seed(self.total_iterations, timestamp - self.min_machine_timestamp, sample_number)
        for param_name, param in self.model.named_parameters():
            if self.use_lora and 'lora' not in param_name:
                continue
            if debug:
                breakpoint()
            if self.use_different_gpu:
                z = torch.normal(mean=0, std=1, size=param.data.size(), dtype=param.data.dtype).to(param.data.device)
            else:
                if isinstance(param, bnb.nn.Params4bit):
                    param_dequantized = bnb.functional.dequantize_4bit(param, param.quant_state, quant_type=self.quant_type)
                    z = torch.normal(mean=0, std=1, size=param_dequantized.data.size(), device=param_dequantized.data.device, dtype=param_dequantized.data.dtype)
                    param_dequantized.data = param_dequantized.data + scaling_factor * z * self.epsilon
                    param_requantized, param_requantized_state = bnb.functional.quantize_4bit(param_dequantized, quant_type=self.quant_type)
                    param.data = param_requantized.data
                    param.quant_state = param_requantized_state
                else:
                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                    param.data = param.data + scaling_factor * z * self.epsilon

    def get_model_grad(self):
        for param_name, param in self.model.named_parameters():
            if self.use_lora and 'lora' not in param_name:
                continue
            self.grad[param_name] = param.grad.data.clone().cpu()

    def accumulate_grad(self, scaling_factor, timestamp, sample_number):
        set_seed(self.total_iterations, timestamp - self.min_machine_timestamp, sample_number)
        for param_name, param in self.model.named_parameters():
            if self.use_lora and 'lora' not in param_name:
                continue
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if param_name not in self.grad:
                self.grad[param_name] = scaling_factor * z * self.epsilon
            else:
                self.grad[param_name] += scaling_factor * z * self.epsilon

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
                batch = get_batch(self.batch_size, self.dataset['validation'], self.dataset_index, self.target_index)
                loss = self.calculate_loss(self.model, self.tokenizer, batch)
                losses.append(loss.item())
        with open(f'{self.model_name}_full_grad={self.send_full_grad}_normal={self.normal}_{self.min_machine_timestamp}_{self.learning_rate}.txt', 'a+') as f:
            mean_loss = np.mean(losses)
            f.write(self.my_address+": " + str(mean_loss) + " start eval time: " + str(start_round_time - self.timestamp)  +" finish eval time: " + str(time.time() - self.timestamp) +  " iteration: " + str(self.total_iterations ) + " num samples: 10\n")
            wandb.log({
                "machine_address": self.my_address,
                "mean_eval_loss": mean_loss,
                "start_eval_time": start_round_time - self.timestamp,
                "finish_eval_time": time.time() - self.timestamp,
                "iteration": self.total_iterations,
                "num_eval_samples": 10
            })
        return mean_loss

    def update_weights(self):
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
            batch = get_batch(self.batch_size, self.dataset['train'], self.dataset_index, self.target_index)
            if self.normal:
                self.grad["num_samples"] += 1
                loss = self.calculate_loss(self.model, self.tokenizer, batch)
                loss.backward()
                loss = loss.detach()
                self.losses.append(loss.item())
                print(f"Projected gradient: disabled  - time elapsed: {time.time() - init_time}, loss = {loss.item()}")
            else:
                self.perturbed = True
                self.add_perturbation(1, self.timestamp, self.sample_number)
                loss_1 = self.calculate_loss(self.model, self.tokenizer, batch).item()
                self.add_perturbation(-2, self.timestamp, self.sample_number)
                loss_2 = self.calculate_loss(self.model, self.tokenizer, batch).item()
                self.add_perturbation(1, self.timestamp, self.sample_number)
                self.perturbed = False
                self.projected_grads.append((loss_1 - loss_2) / (2 * self.epsilon))
                self.losses.append((loss_1 + loss_2) / 2)
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
        with open(f'{self.model_name}_full_grad={self.send_full_grad}_normal={self.normal}_{self.min_machine_timestamp}_{self.learning_rate}.txt', 'a+') as f:
            mean_loss = np.mean(self.losses)
            f.write(f'{self.my_address}: train {mean_loss} start inference time: {start_round_time - self.timestamp} finish inference time: {time.time() - self.timestamp} iteration: {self.total_iterations} num samples: {self.sample_number}\n')
            wandb.log({
                "mode": "train",
                "machine_address": self.my_address,
                "mean_inference_loss": mean_loss,
                "start_inference_time": start_round_time - self.timestamp,
                "finish_inference_time": time.time() - self.timestamp,
                "iteration": self.total_iterations,
                "num_inference_samples": self.sample_number
            })

    def evaluate_model(self):
        self.sample_number = 0
        self.losses = []
        self.model.eval()

        start_round_time = time.time()
        while self.sample_number < self.gradient_acc_steps and  (time.time() - start_round_time) < self.increment_time:
            init_time = time.time()
            print(f"Sample number: {self.sample_number} - inference time remaining: {self.increment_time + start_round_time - time.time() }")
            while self.sending_weights:
                time.sleep(0.1)
            batch = get_batch(self.batch_size, self.dataset['train'], self.dataset_index, self.target_index)

            with torch.no_grad():
                loss = self.calculate_loss(self.model, self.tokenizer, batch)
                self.losses.append(loss.item())
            print(f"Time elapsed: {time.time() - init_time}, loss = {loss.item()}")

            # If the elapsed time is greater than the inference time, warn the user
            if time.time() - init_time > self.inference_time:
                print("Warning: updating preset inference time to the actual inference time.")
                self.inference_time = time.time() - init_time
            self.sample_number += 1

        if self.sample_number < self.gradient_acc_steps:
            print("Warning: did not have enough samples to reach desired inference steps.")

        # Write mean loss to loss.txt
        with open(f'{self.model_name}_full_grad={self.send_full_grad}_normal={self.normal}_{self.min_machine_timestamp}_{self.learning_rate}.txt', 'a+') as f:
            mean_loss = np.mean(self.losses)
            f.write(f'{self.my_address}: evaluate {mean_loss} start inference time: {start_round_time - self.timestamp} finish inference time: {time.time() - self.timestamp} iteration: {self.total_iterations} num samples: {self.sample_number}\n')
            wandb.log({
                "mode": True,
                "machine_address": self.my_address,
                "mean_inference_loss": mean_loss,
                "start_inference_time": start_round_time - self.timestamp,
                "finish_inference_time": time.time() - self.timestamp,
                "iteration": self.total_iterations,
                "num_inference_samples": self.sample_number
            })

    def request_grads_from_all_machines(self):
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

    def apply_all_grads(self, all_projected_grads):
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
                self.apply_full_grad(-self.learning_rate / num_samples, all_projected_grads[address])
        else:
            num_samples = sum([len(all_projected_grads[address]) for address in sorted_addresses])
            for address in sorted_addresses:
                address_grads = all_projected_grads[address]
                for grad_idx, grad in enumerate(address_grads):
                    if self.debug:
                        breakpoint()
                    self.add_perturbation(
                        -self.learning_rate * grad / num_samples,
                        self.addresses_timestamp[address], grad_idx, self.debug)
        self.perturbed = False
        with open(f'{self.model_name}_full_grad={self.send_full_grad}_normal={self.normal}_{self.min_machine_timestamp}_{self.learning_rate}.txt', 'a+') as f:
            mean_loss = np.mean(self.losses)
            f.write(f'{self.my_address}: apply {mean_loss} start grad time: {start_round_time - self.timestamp} finish grad time: {time.time() - self.timestamp} iteration: {self.total_iterations} num samples: {num_samples}\n')
            wandb.log({
                "machine_address": self.my_address,
                "mean_grad_loss": mean_loss,
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
        while (len(self.num_finish[description]) < count) and (time.time() - start_time < timeout): # exclude the current machine
            print(f"Waiting for machines to {description}... currently {len(self.num_finish[description]) + 1}/{count + 1}")
            time.sleep(0.1)
        if len(self.num_finish[description]) < count:
            if quit_on_timeout:
                print(f"Timed out waiting for machines to finish {description}.")
                sys.exit()
            missing_addresses = set(self.all_addresses) - self.num_finish[description]
            print(f"Warning: {missing_addresses} did not {description} in time, removing them from the list of machines.")
            self.all_addresses = list(set(self.all_addresses) - self.num_finish[description])
        self.num_finish[description] = set()

    def sync(self, description="initialize", count=None):
        if count is None:
            count = len(self.all_addresses)
        self.notify_finish(description)
        self.wait_till_finish(count, description)

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
                if self.use_backup:
                    print("Backing up weights.")
                    self.backup_weights = {k: v.cpu() for k, v in self.model.state_dict().items()}
                print("Calculating losses.")
                self.update_weights()
                self.sync("finish forward pass")
                if self.use_backup:
                    print("Restoring weights.")
                    self.model.load_state_dict(self.backup_weights)
                print("Requesting gradients.")
                all_projected_grads = self.request_grads_from_all_machines()
                print("Applying gradients.")
                self.apply_all_grads(all_projected_grads)
                self.total_iterations += 1
                print("Evaluating model.")
                self.evaluate_model()
                self.sync("finish applying gradients")
                print(f"Finished training for iteration {self.total_iterations} ending at", time.time())
                if self.all_addresses:  # Choose a random address to check the hash
                    self.model = confirm_hash(np.random.choice(self.all_addresses), self.model)
                if self.timestamp == self.min_machine_timestamp:
                    if self.total_iterations % self.eval_interval == 0:
                        self.eval()

        self.sync("exit")

def calculate_loss(model, tokenizer, batch):
    tokenized_batch = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, return_token_type_ids=False if "llama" in tokenizer.name_or_path else None)
    device = next(model.parameters()).device
    tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}
    outputs = model(**tokenized_batch, labels=tokenized_batch["input_ids"])
    loss = outputs.loss
    del outputs
    return loss

def calculate_conditional_loss(model, tokenizer, batch_tuple):
    batch_in, batch_out = batch_tuple
    # Concatenate the strings
    batch = [in_str + out_str for in_str, out_str in zip(batch_in, batch_out)]
    tokenized_batch = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, return_token_type_ids=False if "llama" in tokenizer.name_or_path else None)
    device = next(model.parameters()).device
    tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}
    
    outputs = model(**tokenized_batch, labels=tokenized_batch["input_ids"])

    labels = tokenized_batch["input_ids"].clone()
    logits = outputs.logits
    # Ignore batch_in tokens
    for i, in_str in enumerate(batch_in):
        labels[i, :len(tokenizer.encode(in_str))] = -100
    # Ignore padding tokens
    labels[labels == tokenizer.pad_token_id] = -100
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss

def get_batch(batch_size, dataset, dataset_index, dataset_target_index=None):
    # Randomly choose indices for batch sampling
    indices = np.random.choice(len(dataset), size=batch_size)
    batch = dataset[indices]  # Select the batch from the dataset
    if dataset_target_index is None:
        batch = batch[dataset_index]
        batch = [text for text in batch if text.strip()]  # Filter empty strings
    else:
        batch_in = batch[dataset_index]
        target = batch[dataset_target_index]
        target_feature = dataset.features[dataset_target_index]
        if isinstance(target_feature, datasets.ClassLabel):
            target_map = target_feature.int2str
        else:
            target_map = lambda x: str(x)
        batch = list(zip(*[(str(text + "\n"), target_map(target)) for text, target in zip(batch_in, target) if text.strip()]))
    if len(batch) < 3 and batch_size >= 3 and dataset_target_index is None:  # If the batch size doesn't match, try again
        return get_batch(batch_size, dataset, dataset_index, dataset_target_index)
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

def calculate_hash(model, decimals=3):
    # Calculates the hash of the model to verify the model is the same on all machines
    # Ignore differences on the order of epsilon, so we round
    str_model = ""
    for _, param in model.named_parameters():
        if isinstance(param, bnb.nn.Params4bit):
            param_dequantized = bnb.functional.dequantize_4bit(param, param.quant_state, quant_type=self.quant_type)
            str_model += str((param_dequantized.data * 10 ** decimals).round().int().cpu().numpy())
        else:
            str_model += str((param.data * 10 ** decimals).round().int().cpu().numpy())
    serialized_model = str_model.encode()
    model_hash = hashlib.md5(serialized_model).hexdigest()
    print(f"Model hash: {model_hash}")
    return model_hash

def confirm_hash(address, model):
    try:
        response = requests.get(f"{address}/hash")
    except:  # If the hash don't match, the model is reset
        print("Error: No response from address, assuming hash is correct.")
    if calculate_hash(model) != response.json()['hash']:
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
    parser.add_argument('--increment_time', type=float, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--max_iterations', type=int, default=300)
    parser.add_argument('--gradient_acc_steps', type=float, default=30)
    parser.add_argument('--buffer_time', type=float, default=0)
    parser.add_argument('--inference_time', type=float, default=1)
    parser.add_argument('--min_num_machines', type=int, default=2)
    parser.add_argument('--send_full_grad', type=bool, default=False)
    parser.add_argument('--normal', type=bool, default=False)
    parser.add_argument('--use_different_gpu', type=bool, default=False)
    parser.add_argument('--use_bnb', type=bool, default=False)
    parser.add_argument('--start_ip', type=str, default= "127.0.0.1")
    parser.add_argument('--self_ip', type=str, default= "127.0.0.1")
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--conditional', type=bool, default=True)
    
    args = parser.parse_args()
    server = Machine(f'http://{args.self_ip}:{args.port}', [f'http://{args.start_ip}:7000'], args.increment_time, args.buffer_time, args.inference_time, epsilon=args.epsilon, batch_size=args.batch_size, model_name=args.model_name, min_num_machines=args.min_num_machines, send_full_grad=args.send_full_grad, normal=args.normal, use_different_gpu=args.use_different_gpu, debug=args.debug, gradient_acc_steps=args.gradient_acc_steps, learning_rate=args.learning_rate, max_iterations=args.max_iterations, dataset_name=args.dataset_name, dataset_index=args.dataset_index, use_bnb=args.use_bnb, conditional=args.conditional)
    wandb.init(project="justonebyte", name = f'{args.model_name}_full_grad={args.send_full_grad}_normal={args.normal}_{server.timestamp}_{args.self_ip}_{args.learning_rate}')

    #save config
    # with open(f'config_{args.self_ip}_{server.timestamp}.json', 'w') as f:
    #     json.dump(vars(args), f)
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