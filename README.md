# Graph Neural Networks
I implemented a variety of algorithms to perform transductive graph classification on the [Cora dataset](https://github.com/tkipf/pygcn/tree/master/data/cora).

The full report can be found [here](report.pdf)

## Implemented algorithms
- Simple SVM baseline using node features alone
- SVM using node features concatenated with average node features of the neighbour nodes
- Multilayer perceptron using node features concatenated with average node features of the neighbour nodes
- Graph Convolutional Neural Network (GCN) [1]
- Graph Attention Networks (GAT) [2]


## References
[1] Kipf, Thomas N., and Max Welling. "Semi-supervised classification with graph convolutional networks." arXiv preprint arXiv:1609.02907 (2016).

[2] Veličković, Petar, et al. "Graph Attention Networks." arXiv preprint arXiv:1710.10903 (2017).
