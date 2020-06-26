# Deep Learning Medical Imagin Project: Tel-Aviv Universty


## Setup

This is python3.6 and Pytorch based code. Dependencies:

```
 pip install -r requirements.txt
```

## Datasets

Datasets as tarballs are available from the links below.

- [Rotated MNIST](http://bergerlab-downloads.csail.mit.edu/spatial-vae/mnist_rotated.tar.gz)
- [5HDB simulated EM images](http://bergerlab-downloads.csail.mit.edu/spatial-vae/5HDB.tar.gz)
- [CODH/ACS EM images](http://bergerlab-downloads.csail.mit.edu/spatial-vae/codhacs.tar.gz)
- [Antibody EM images](http://bergerlab-downloads.csail.mit.edu/spatial-vae/antibody.tar.gz)

Download and extract. Working directory stracture:

```
.
├── LICENSE
├── README.md
├── configs
│   └── vae_mnist.yaml
├── data
│   └── mnist_rotated
├── externals
│   └── spatial_vae
├── models
├── output
├── requirements.txt
└── src
```




## Usage

Training spatial-VAE model:

```
cd src
python main_train_vae.py --config_path=../configs/vae_mnist.yaml
python main_train_vae.py --config_path=../configs/vae_5hdb.yaml
```

configuration file is located here: ''configs/vae_mnist.yaml"



Training our approach

```
cd src
python main_train_ours.py --config_path=../configs/ours_mnist.yaml 
python main_train_ours.py --config_path=../configs/ours_5hdb.yaml
```

configuration file is located here: ''configs/ours_mnist.yaml"



To execute the original code:

```bash
cd externals/spatial_vae
python train_mnist.py --no-translate --minibatch-size 128
```



## License

This source code is provided under the [MIT License](https://github.com/tbepler/spatial-VAE/blob/master/LICENSE).

