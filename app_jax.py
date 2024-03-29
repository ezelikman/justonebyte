import numpy as np
import time
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from argparse import ArgumentParser
import torch
torch.use_deterministic_algorithms(True)
from flask import Flask, request, send_file, Response
from threading import Thread
import hashlib
import pickle
from flax.core.frozen_dict import freeze, unfreeze
import jax
import jax.numpy as jnp
import jax.nn
import math

class Machine:
    def __init__(self, my_address, initial_server_addresses,
                increment_time, buffer_time, inference_time,
                epsilon=0.001, batch_size=16, use_backup=True,
                model_name='gpt2', dataset_name=('gsm8k', 'main'), dataset_index='question',
                device='best', dtype=torch.float16, use_lora=False, min_num_machines=2, send_full_grad=False,
                normal=False, use_different_gpu=False, debug=False, gradient_acc_steps=1, learning_rate=1e-1,
                backend='jax'
        ):
        self.backend = backend
        self.add_perturbation = self.add_perturbation_jax if self.backend == 'jax' else self.add_perturbation_torch
        self.my_address = my_address
        self.dataset_name = dataset_name  # Name of the dataset to be used
        self.dataset_index = dataset_index  # Index of the dataset to be used
        self.buffer_time = buffer_time  # Time to stop inferencing before end_time
        self.epsilon = epsilon  # Perturbation size
        if device == 'best':
            if backend != 'jax':
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = jax.devices('gpu')[0]
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
        self.num_finish = {
            "finish initialize model": 0,
            "finish forward pass": 0,
            "finish applying gradients": 0
        }
        self.use_different_gpu = use_different_gpu
        self.debug = debug
        self.gradient_acc_steps=gradient_acc_steps
        self.max_iterations = 300
        self.learning_rate=learning_rate # learning rate for the optimizer; will be overwritten by the main machine if it is not the main machine
        self.app = Flask(__name__)

        def calculate_tokenized_loss_jax(input_ids, attention_mask):
            # Convert tokenized_batch to JAX arrays
            outputs = self.model(input_ids, attention_mask)
            logits = outputs.logits
            labels = input_ids
            loss = compute_loss(logits, labels)
            return loss

        def calculate_loss_jax(model, tokenizer, batch):
            reference_batch = tokenizer(batch, padding=True, truncation=True, return_token_type_ids=False if "llama" in tokenizer.name_or_path else None, return_tensors='jax')
            max_length = reference_batch['input_ids'].shape[-1]
            max_length_power_of_two = 2**math.ceil(math.log(max_length, 2))
            tokenized_batch = tokenizer(batch, padding='max_length', truncation=True, return_token_type_ids=False if "llama" in tokenizer.name_or_path else None, return_tensors='jax', max_length=max_length_power_of_two)
            return self.calculate_tokenized_loss_jax(tokenized_batch['input_ids'], tokenized_batch['attention_mask'])

        self.calculate_tokenized_loss_jax = jax.jit(calculate_tokenized_loss_jax)
        self.calculate_loss = calculate_loss_jax if self.backend == "jax" else calculate_loss_torch

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
            self.num_finish[description] += 1
            return {'num_finish': self.num_finish[description]}, 200

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

    ### Initialization functions ###
    def initialize_run(self):
        self.initialize_model()

    def initialize_model(self, use_default=False):
        if use_default or self.total_iterations == 0:
            print("Initializing default model")
            if self.backend == 'jax':
                from transformers import FlaxAutoModelForCausalLM as AutoModelForCausalLM
            else:
                from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name
            )
            if self.backend != 'jax':
                self.model.eval()
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
        resize_token_embeddings = resize_token_embeddings_jax if self.backend == 'jax' else resize_token_embeddings_torch
        resize_token_embeddings(self.model, len(self.tokenizer))
        if not use_default and self.total_iterations != 0:
            self.model.load_state_dict(torch.load('received_model.pt'))
            print("Loaded model from", random_address)
        if self.model.config.pad_token_id == -1:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = model_processing(self.model, self.dtype, self.device, self.use_lora, self.backend)

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
    def add_perturbation_jax(self, scaling_factor, timestamp, sample_number, debug=False):
        self.key = set_seed_jax(self.total_iterations, timestamp - self.min_machine_timestamp, sample_number)
        self.inferenced = False
        def add_noise(param):
            self.key, subkey = jax.random.split(self.key)
            noise = jax.random.normal(subkey, param.shape)
            if not self.inferenced:
                print(noise.ravel()[-1])
                self.inferenced = True
            return param + noise * self.epsilon * scaling_factor
        self.model.params = jax.tree_map(add_noise, self.model.params)

    def add_perturbation_torch(self, scaling_factor, timestamp, sample_number, debug=False):
        set_seed_torch(self.total_iterations, timestamp - self.min_machine_timestamp, sample_number)
        for param_name, param in self.model.named_parameters():
            if self.use_lora and 'lora' not in param_name:
                continue
            if debug:
                breakpoint()
            if self.use_different_gpu:
                z = torch.normal(mean=0, std=1, size=param.data.size(), dtype=param.data.dtype).to(param.data.device)
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.epsilon

    def get_model_grad(self):
        for param_name, param in self.model.named_parameters():
            if self.use_lora and 'lora' not in param_name:
                continue
            self.grad[param_name] = param.grad.data.clone()

    def accumulate_grad(self, scaling_factor, timestamp, sample_number):
        set_seed_torch(self.total_iterations, timestamp - self.min_machine_timestamp, sample_number)
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
            param.data = param.data + scaling_factor * grad[param_name]

    def update_weights(self):
        self.grad = {"num_samples": 0}
        self.projected_grads = []
        self.losses = []
        self.sample_number = 0
        if self.normal:
            self.model.zero_grad()
        start_round_time = time.time()
        while self.sample_number < self.gradient_acc_steps and  (time.time() - start_round_time) < self.increment_time:
            init_time = time.time()
            print(f"Sample number: {self.sample_number} - inference time remaining: {self.increment_time + start_round_time - time.time() }")
            while self.sending_weights:
                time.sleep(0.1)
            batch = get_batch(self.batch_size, self.dataset, self.dataset_index)
            if self.normal:
                self.grad["num_samples"] += 1
                loss = self.calculate_loss_torch(self.model, self.tokenizer, batch)
                loss.backward()
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
        model_name = self.model_name.split('/')[-1]
        with open(f'{model_name}_full_grad={self.send_full_grad}_normal={self.normal}_{self.min_machine_timestamp}.txt', 'a+') as f:
            f.write(self.my_address+": " + str(np.mean(self.losses)) + " start inference time: " + str(start_round_time - self.timestamp)  +" finish inference time: " + str(time.time() - self.timestamp) +  " iteration: " + str(self.total_iterations ) + '\n')

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
        model_name = self.model_name.split('/')[-1]
        with open(f'{model_name}_full_grad={self.send_full_grad}_normal={self.normal}_{self.min_machine_timestamp}.txt', 'a+') as f:
            f.write(self.my_address+": " + str(np.mean(self.losses)) + " start grad time: " + str(start_round_time - self.timestamp)  +" finish grad time: " + str(time.time() - self.timestamp) +  " iteration: " + str(self.total_iterations ) + '\n')

    def notify_finish(self, description="initialize"):
        for address in self.all_addresses:
            try:
                response = requests.post(f"{address}/notify_finish" , json={'address': self.my_address, 'description': description})
            except Exception as e:
                print(e)
                print(f"Error: could not connect to {address}")
                continue

    def wait_till_finish(self, count=None, description="initialize"):
        # wait for everyone to finish
        if count is None:
            count = len(self.all_addresses)
        while self.num_finish[description] < count: # exclude the current machine
            print(f"Waiting for machines to {description}... currently {self.num_finish[description] + 1}/{count + 1}")
            time.sleep(0.1)
        self.num_finish[description] = 0

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
                    if self.backend == 'jax':
                        self.backup_weights = jax.device_put(self.model.params, device=jax.devices('cpu')[0])
                    else:
                        self.backup_weights = {k: v.cpu() for k, v in self.model.state_dict().items()}
                print("Calculating losses.")
                self.update_weights()
                self.sync("finish forward pass")
                if self.use_backup:
                    print("Restoring weights.")
                    if self.backend == 'jax':
                        gpu_devices = [device for device in jax.devices() if 'gpu' in str(device).lower()]
                        self.model.params = jax.device_put(self.backup_weights, device=gpu_devices[0])
                    else:
                        self.model.load_state_dict(self.backup_weights)
                print("Requesting gradients.")
                all_projected_grads = self.request_grads_from_all_machines()
                print("Applying gradients.")
                self.apply_all_grads(all_projected_grads)
                self.sync("finish applying gradients")
                print("Finished training for this round ending at", time.time())
                if self.all_addresses:  # Choose a random address to check the hash
                    self.model = confirm_hash(np.random.choice(self.all_addresses), self.model)
                self.total_iterations += 1

@jax.jit
def compute_loss(logits, labels):
    ignore_index = -100
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    log_probs = jnp.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)
    loss = jnp.where(labels != ignore_index, log_probs, 0)
    loss = -jnp.sum(loss) / jnp.sum(labels != ignore_index)
    return loss

def calculate_loss_torch(model, tokenizer, batch):
    tokenized_batch = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, return_token_type_ids=False if "llama" in tokenizer.name_or_path else None)
    device = next(model.parameters()).device
    tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}
    outputs = model(**tokenized_batch, labels=tokenized_batch["input_ids"])
    loss = outputs.loss
    return loss

def get_batch(batch_size, dataset, dataset_index):
    # Randomly choose indices for batch sampling
    indices = np.random.choice(len(dataset), size=batch_size)
    batch = dataset[indices]  # Select the batch from the dataset
    batch = batch[dataset_index]
    batch = [text for text in batch if text.strip()]  # Filter empty strings
    if len(batch) < 3:  # If the batch size doesn't match, try again
        return get_batch(batch_size, dataset, dataset_index)
    return batch

def model_processing(model, dtype, device, use_lora, backend='pt'):
    if backend != 'jax':
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

def set_seed_jax(total_iterations, timestamp, sample_number):
    timer = int(f'{timestamp:.6f}'.replace('.', ''))
    # Handle C-long overflow
    full_seed = int(f'{total_iterations}{sample_number:03}{timer}') % 9223372036854775806
    return jax.random.PRNGKey(full_seed)

def set_seed_torch(total_iterations, timestamp, sample_number):
    timer = int(f'{timestamp:.6f}'.replace('.', ''))
    full_seed = int(f'{total_iterations}{sample_number:05}{timer}') % 9223372036854775806
    torch.manual_seed(full_seed)
    torch.cuda.manual_seed(full_seed)

def resize_token_embeddings_jax(model, new_size):
    if model.config.vocab_size == new_size:
        return
    model.config.vocab_size = new_size
    params = model.params
    rnd_key = jax.random.PRNGKey(0)
    params = unfreeze(params)
    old_embeddings = params['transformer']['wte']['embedding']
    old_size = old_embeddings.shape[0]
    dim = old_embeddings.shape[1]
    initializer = jax.nn.initializers.normal(stddev=model.config.initializer_range)
    new_embeddings = initializer(rnd_key, (new_size, dim))
    new_embeddings = new_embeddings.at[:old_size].set(old_embeddings)
    params['transformer']['wte']['embedding'] = new_embeddings
    params = freeze(params)
    model.params = params

def resize_token_embeddings_torch(model, new_size):
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
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epsilon', type=float, default=1e-3)
    parser.add_argument('--increment_time', type=float, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--gradient_acc_steps', type=float, default=30)
    parser.add_argument('--buffer_time', type=float, default=0)
    parser.add_argument('--inference_time', type=float, default=1)
    parser.add_argument('--min_num_machines', type=int, default=2)
    parser.add_argument('--send_full_grad', type=bool, default=False)
    parser.add_argument('--normal', type=bool, default=False)
    parser.add_argument('--use_different_gpu', type=bool, default=False)
    parser.add_argument('--start_ip', type=str, default= "127.0.0.1")
    parser.add_argument('--self_ip', type=str, default= "127.0.0.1")
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    server = Machine(f'http://{args.self_ip}:{args.port}', [f'http://{args.start_ip}:7000'], args.increment_time, args.buffer_time, args.inference_time, epsilon=args.epsilon, batch_size=args.batch_size, model_name=args.model_name, min_num_machines=args.min_num_machines, send_full_grad=args.send_full_grad, normal=args.normal, use_different_gpu=args.use_different_gpu, debug=args.debug, gradient_acc_steps=args.gradient_acc_steps, learning_rate=args.learning_rate)
    Thread(target=server.start_server, args=(args.port,)).start()
    server.run()
