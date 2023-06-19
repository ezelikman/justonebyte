# JustOneByte

Repo for [Just One Byte (per gradient): A Note on Low-Bandwidth Decentralized Language Model Finetuning Using Shared Randomness](https://arxiv.org/abs/2306.10015).

To launch a training run across multiple machines, run `app.py` with the arguments `start_ip` corresponding to the IP of a known machine, and `self_ip` corresponding to the IP by which other machines should communicate with the current machine. `learning_rate` specifies learning rate and `model_name` specifies a HuggingFace Transformers-supported model. `gradient_acc_steps` specifies the minimum number of gradients each machine should calculate before sharing the gradients with the others and updating their weights. You can also use `port` to specify the current machine's port and `start_port` to specify a known machine's port.
