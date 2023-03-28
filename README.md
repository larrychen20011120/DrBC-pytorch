# DrBC-pytorch
The implementation of paper named "Learning to Identify High Betweenness Centrality Nodes from Scratch: A Novel Graph Neural Network Approach" (CIKM 2019).

## How to use it
### generating your own data
```
python generator.py --batch_size 16 --train_size 10000 --eval_size 100 --scale 0 1 2 3 4
```
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
