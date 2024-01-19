
# Partially Supervised UMAP with Transformer Encoder

A repository for the reproduction of results in the paper "Partially-Supervised Metric Learning via Dimensionality Reduction of Text Embeddings using Transformer Encoders and Attention Mechanisms", for the IEEE Access Journal.




## Installation

To run the experiments, clone this repo and then download and unzip the associated data, and resources folders, which can be found here:

Install the requirements:
```bash
  pip install -r requirements.txt
```
Note: It is strongly advised to run this experiment on a GPU using tensorflow. The version of tensorflow best suited to you may be different, depending on the version of the CUDA toolkit which you have installed on your local machine. You can check which version of TF best suits your CUDA version here:
https://www.tensorflow.org/install/source#gpu
## Usage
Then, to generate the clustering results, and walltimes for each experiment:

```python3 main.py```

To generate the base-line clustering results, for K-Means without Dimensionality Reduction Applied:

```python3 kmeans_baseline.py```

If you would like to use the Transformer Encoder architecture for your own experiments, you can copy the code from ```encoder.py``` and use the ```build_model()``` function, providing your own hyperparameters. This can then be directly passed to parametric UMAP as an encoder model (see ```generate.py``` lines 97-118).

## This paper is built upon the following works:

```
@article{mcinnes2018umap,
  title={Umap: Uniform manifold approximation and projection for dimension reduction},
  author={McInnes, Leland and Healy, John and Melville, James},
  journal={arXiv preprint arXiv:1802.03426},
  year={2018}
}

 @article{sainburg2021parametric,
  title={Parametric UMAP Embeddings for Representation and Semisupervised Learning},
  author={Sainburg, Tim and McInnes, Leland and Gentner, Timothy Q},
  journal={Neural Computation},
  volume={33},
  number={11},
  pages={2881--2907},
  year={2021},
  publisher={MIT Press One Rogers Street, Cambridge, MA 02142-1209, USA journals-info~…}
}

@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```