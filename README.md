# DrBC-pytorch
The implementation of paper named "Learning to Identify High Betweenness Centrality Nodes from Scratch: A Novel Graph Neural Network Approach" (CIKM 2019).

## Project structure
* **model.py**: the model structure of DrBC
* **utils.py**: the loss and metrics for DrBC training and testing
* **train.py**: define how to train the model DrBC
* **test.py**: given path of graph and ground truth, and then calculate the testing result
* **generator.py**: definition of dataset and how to get information from it, and it also contains the main function to generate the training data, which will store in **train_val_gen/**
* **test_30/**: it is a directory for generating the final testing synthetic data (nodes: 5000, 10000, 20000, 50000, 100000)
* **test.sh**: shell script for testing all TA's synthetic dataset
* **reproduce.ipynb**: notebook for reproducing the paper's tables and figures and some results will store in **reproduce/** directory 

## How to use it
### generating your own data
```
python generator.py --batch_size 16 --train_size 10000 --eval_size 100 --scale 0 1 2 3 4
```
if you want to use the data I generated: click [here](https://drive.google.com/file/d/1Zb2HiBhDZVEtHFHrM4QipIlghBlWD1Cs/view?usp=share_link)
### for training
```
python train.py --scale 100 200 --optimizer adam --lr 1e-4
```
### for testing
* single file
example for testing the com-youtube dataset with GPU
```
python test.py --graph_path hw1_data/youtube/com-youtube.txt \
 --bc_path hw1_data/youtube/com-youtube_score.txt \
--model_path model_result/1000_1200.pth --device gpu
```
* hw1_data (multiple files)
example for testing all hw1 synthetic 5000 nodes data (and the result will stored in **result.txt**)
```
rm result.txt
bash test.sh
```

## Baseline Comparison
* Node2Vec: [PyG Node2Vec](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/node2vec.html)
* kadraba: [Networkit](https://networkit.github.io/)
* RK: [Networkit](https://networkit.github.io/)

## Result of my reproduction
### PCA plotting and graph
![](/reproduce/figure4.png)
### Training procedure
![](/reproduce/figure3.png)
