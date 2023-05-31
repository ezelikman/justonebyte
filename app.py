import numpy as np
import time
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from argparse import ArgumentParser
import torch
torch.use_deterministic_algorithms(True)
from flask import Flask, request, send_file
from threading import Thread
import hashlib
from peft import get_peft_model, LoraConfig, TaskType

class Machine:
    def __init__(self, my_address, initial_server_addresses,
                increment_time, buffer_time, inference_time,
                epsilon=0.001, batch_size=16, use_backup=True,
                model_name='gpt2', dataset_name=('gsm8k', 'main'), dataset_index='question',
                device='best', dtype=torch.float16, use_lora=False, min_num_machines=2,
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
        self.inference_time = inference_time  # Time per inference, should be an upper bound
        self.all_addresses = initial_server_addresses  # Addresses of all known servers
        self.min_num_machines = min_num_machines  # Minimum number of machines to train with
        if my_address in self.all_addresses:
            self.all_addresses.remove(my_address)
        self.addresses_timestamp = {self.my_address: self.timestamp}  # Timestamps of all known servers
        self.dataset = load_dataset(*self.dataset_name)['train']  # Initialize the dataset
        self.model_name = model_name  # Name of the model to be used
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # Tokenizer for the model
        if self.tokenizer.pad_token is None:  # Add a padding token if there isn't one - common
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.perturbed = False  # Whether the model has been perturbed (flag to prevent sending weights while perturbed)
        self.sending_weights = False  # Whether the model is currently sending weights (flag to prevent perturbing while sending)
        # We can prevent floating point errors from repeated perturbations by using a backup model
        self.use_backup = use_backup
        self.backup_weights = None
        self.total_iterations = 0
        self.app = Flask(__name__)

        @self.app.route('/model', methods=['GET'])
        def get_model():
            while self.perturbed:
                time.sleep(0.1)
            if self.model is not None:
                self.sending_weights = True
                torch.save(self.model.state_dict(), 'temp_model.pt')
                self.sending_weights = False
                return send_file('temp_model.pt', as_attachment=True)
            else:
                return {'error': 'Model not initialized'}, 500
        
        @self.app.route('/hash', methods=['GET'])
        def get_hash():
            while (time.time() < self.end_time - self.buffer_time) or self.perturbed:
                time.sleep(0.1)
            return {'hash': calculate_hash(self.model)}

        @self.app.route('/grads', methods=['GET'])
        def get_grads():
            if self.model is not None:
                return {'grads': self.projected_grads}
            else:
                return {'error': 'Model not initialized'}, 500

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
                        requests.post(f"{address}/notify", json={'address': addr, 'timestamp': ts})
            return {'all_addresses': self.all_addresses, 'timestamps': self.addresses_timestamp, 'end_time': self.end_time, 'learning_rate': self.learning_rate, 'total_iterations': self.total_iterations}, 200

    ### Initialization functions ###
    def initialize_run(self):
        self.initialize_model(use_default=True)
        self.end_time = time.time() + self.increment_time
        self.learning_rate = 1e-1

    def initialize_model(self, use_default=False):
        if use_default or self.total_iterations == 0:
            print("Initializing default model")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name
            ).eval()
        else:
            random_address = np.random.choice(self.all_addresses)
            print("Getting model from", random_address)
            response = requests.get(f"{random_address}/model")
            print("Received model from", random_address)
            with open('received_model.pt', 'wb') as f:
                f.write(response.content)
            # Load from pretrained config
            config = AutoConfig.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_config(config).eval()
        resize_token_embeddings(self.model, len(self.tokenizer))
        if not use_default and self.total_iterations != 0:
            self.model.load_state_dict(torch.load('received_model.pt'))
            print("Loaded model from", random_address)
        if self.model.config.pad_token_id == -1:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = model_processing(self.model, self.dtype, self.device, self.use_lora)

    ### Server functions ###
    def start_server(self, port):
        self.app.run(port=port)

    def announce_existence(self):
        for address in self.all_addresses:
            response = requests.post(f"{address}/notify", json={'address': self.my_address, 'timestamp': self.timestamp}).json()
            for addr in response['all_addresses']:
                if addr not in self.all_addresses and addr != self.my_address:
                    self.all_addresses.append(addr)
            self.addresses_timestamp.update(response['timestamps'])
            if self.end_time is None:
                self.end_time = response['end_time']
            self.learning_rate = response['learning_rate']
            self.total_iterations = response['total_iterations']

    ### Training functions ###
    def add_perturbation(self, scaling_factor, timestamp, sample_number):
        set_seed(self.end_time, timestamp, sample_number)
        for param_name, param in self.model.named_parameters():
            if self.use_lora and 'lora' not in param_name:
                continue
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.epsilon

    def update_weights(self):
        self.projected_grads = []
        self.losses = []
        self.sample_number = 0
        while time.time() < self.end_time - self.buffer_time - self.inference_time:
            init_time = time.time()
            print(f"Sample number: {self.sample_number} - inference time remaining: {self.end_time - init_time - self.buffer_time - self.inference_time}")
            while self.sending_weights:
                time.sleep(0.1)
            batch = get_batch(self.batch_size, self.dataset, self.dataset_index)
            self.perturbed = True
            self.add_perturbation(1, self.timestamp, self.sample_number)
            loss_1 = calculate_loss(self.model, self.tokenizer, batch)
            self.add_perturbation(-2, self.timestamp, self.sample_number)
            loss_2 = calculate_loss(self.model, self.tokenizer, batch)
            self.add_perturbation(1, self.timestamp, self.sample_number)
            self.perturbed = False
            self.projected_grads.append((loss_1 - loss_2) / (2 * self.epsilon))
            self.losses.append((loss_1 + loss_2) / 2)
            print(f"Projected gradient: {self.projected_grads[-1]} - time elapsed: {time.time() - init_time}, loss = {(loss_1 + loss_2) / 2}")
            # If the elapsed time is greater than the inference time, warn the user
            if time.time() - init_time > self.inference_time:
                print("Warning: updating preset inference time to the actual inference time.")
                self.inference_time = time.time() - init_time
            self.sample_number += 1
        # Write mean loss to loss.txt
        with open('loss.txt', 'a+') as f:
            f.write(str(np.mean(self.losses)) + '\n')
        while time.time() < self.end_time - self.buffer_time:
            time.sleep(0.1)

    def request_grads_from_all_machines(self):
        all_projected_grads = {
            self.my_address: self.projected_grads
        }
        for address in self.all_addresses:
            response = requests.get(f"{address}/grads").json()
            if not 'grads' in response:
                print(f"Error: {response['error']}")
                continue
            all_projected_grads[address] = response['grads']
        return all_projected_grads

    def apply_all_grads(self, all_projected_grads):
        while self.sending_weights:
            time.sleep(0.1)
        self.perturbed = True
        # Get the sorted list of addresses
        sorted_addresses = sorted(all_projected_grads.keys(), key=lambda x: self.addresses_timestamp[x])
        # Calculate the number of samples
        num_samples = sum([len(all_projected_grads[address]) for address in sorted_addresses])
        for address in sorted_addresses:
            address_grads = all_projected_grads[address]
            for grad_idx, grad in enumerate(address_grads):
                self.add_perturbation(
                    -self.learning_rate * grad / num_samples,
                    self.addresses_timestamp[address], grad_idx)
        self.perturbed = False

    def run(self):
        self.announce_existence()
        num_joined = 0
        if not self.all_addresses:
            print("No other machines found.")
            self.initialize_run()
        if self.model is None:
            print("Model not initialized.")
            self.initialize_model()
        while len(self.all_addresses) + 1 < self.min_num_machines: # +1 to include the current machine
            if len(self.all_addresses) + 1 > num_joined:
                num_joined = len(self.all_addresses) + 1
                print(f"Waiting for machines to join... currently {len(self.all_addresses) + 1}/{self.min_num_machines}")
            time.sleep(0.1) # sleep for a while before checking again
            if time.time() > self.end_time - self.buffer_time - self.inference_time:
                self.end_time += self.increment_time
        initialization_time = time.time() + 20
        while time.time() < initialization_time:
            time.sleep(0.1)  # make sure all machines have time to initialize
        while time.time() > self.end_time - self.buffer_time - self.inference_time:
            self.end_time += self.increment_time
        print("Starting with end time", self.end_time)

        while True:  # Run the training loop
            with torch.inference_mode():
                print("Starting run.")
                self.announce_existence()
                if self.model is None:
                    print("Model not initialized.")
                    self.initialize_model()
                if self.use_backup:
                    print("Backing up weights.")
                    self.backup_weights = {k: v.cpu() for k, v in self.model.state_dict().items()}
                print("Starting training.")
                self.update_weights()
                all_projected_grads = self.request_grads_from_all_machines()
                if self.use_backup:
                    print("Restoring weights.")
                    self.model.load_state_dict(self.backup_weights)
                print("Applying gradients.")
                self.apply_all_grads(all_projected_grads)
                print("Finished training for this round ending at", self.end_time)
                if self.all_addresses:  # Choose a random address to check the hash
                    self.model = confirm_hash(np.random.choice(self.all_addresses), self.model)
                self.total_iterations += 1
                self.end_time += self.increment_time

def calculate_loss(model, tokenizer, batch):
    tokenized_batch = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
    device = next(model.parameters()).device
    tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}
    outputs = model(**tokenized_batch, labels=tokenized_batch["input_ids"])
    loss = outputs.loss
    return loss.item()

def get_batch(batch_size, dataset, dataset_index):
    # Randomly choose indices for batch sampling
    indices = np.random.choice(len(dataset), size=batch_size)
    batch = dataset[indices]  # Select the batch from the dataset
    batch = batch[dataset_index]
    batch = [text for text in batch if text.strip()]  # Filter empty strings
    if len(batch) < 3:  # If the batch size doesn't match, try again
        return get_batch(batch_size, dataset, dataset_index)
    return batch

def model_processing(model, dtype, device, use_lora):
    model.to(dtype)  # Process a model after loading it
    model.eval()
    model.to(device)
    if use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=True, r=8, lora_alpha=32, lora_dropout=0.0
        )
        model = get_peft_model(model, peft_config)
    return model

def set_seed(end_time, timestamp, sample_number):
    diff_time = abs(end_time - timestamp) # Convert timestamps to 32-bit integer
    timer = format(diff_time, '.8f').replace('.', '')
    full_seed = int(timer + str(sample_number))  # Sample number is already an int
    torch.manual_seed(full_seed)

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
            print("Hashes don't match.")
            model = None
        else:
            print("Hashes match.")
        return model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--port', type=int, default=7000)
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epsilon', type=float, default=1e-3)
    parser.add_argument('--increment_time', type=float, default=30)
    parser.add_argument('--buffer_time', type=float, default=10)
    parser.add_argument('--inference_time', type=float, default=1)
    parser.add_argument('--min_num_machines', type=int, default=2)
    args = parser.parse_args()
    server = Machine(f'http://127.0.0.1:{args.port}', [f'http://127.0.0.1:7000'], args.increment_time, args.buffer_time, args.inference_time, epsilon=args.epsilon, batch_size=args.batch_size, model_name=args.model_name, min_num_machines=args.min_num_machines)
    Thread(target=server.start_server, args=(args.port,)).start()
    server.run()
