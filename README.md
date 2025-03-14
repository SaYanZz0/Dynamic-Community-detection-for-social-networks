# Deep Temporal Graph Clustering

This is an implementation of deep temporal graph clustering Paper, including a series of related papers and open source datasets.


## Key Paper

### [1] ICLR 2024: Deep Temporal Graph Clustering

Authors: Meng Liu, Yue Liu, Ke Liang, Wenxuan Tu, Siwei Wang, Sihang Zhou, Xinwang Liu

Link: https://arxiv.org/abs/2305.10738


## Code

 You should Note that i used Cpu for Excuting the code because CUDA doesn't support my GPU , so if you want to use CUDA you can change the code .

## Prepare

To run the code, you need prepare datasets and pretrain embeddings:

#### For Datasets

You can download the datasets from [Data4TGC]((https://drive.google.com/drive/folders/1-4O3V0ZcC_f8yP5ylW9CX-lE6qucbFfh)) and create "data" folder in the same directory as the "emb" and "framework" folders.

#### For Pre-Training

In ```./framework/pretrain/```, you need run the ```pretrain.py``` to generate pretrain embeddings.

Note that these embeddings are used for TGC training, while the features in the dataset are used for training by any other method.

That is, the pre-training of node2vec is only part of the TGC.

#### For Training

You need create a folder for each dataset in ```./emb/``` to store generated node embeddings.

For example, after training with `Patent` dataset, the node embeddings will be stored in ```./emb/patent/```


### Run

For each dataset, create a folder in ```emb``` folder with its corresponding name to store node embeddings, i.e., for arXivAI dataset, create ```./emb/arXivAI```.

For training, run the ```main.py``` in the ```./framework``` folder, all parameter settings have default values, you can adjust them in ```main.py```.

### Test

For test, you have two ways:

(1) In the training process, we evaluate the clustering performance for each epoch.

(2) You can also run the ```clustering.py``` in the ```./framework/experiments``` folder.

Note that the node embeddings in the ```./emb./patent/patent_TGC_200.emb``` folder are just placeholders, you need to run the main code to generate them.

