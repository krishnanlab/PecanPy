# Efficient node2vec
A fast, parallelized, memory efficient, and cache optimized Python implementation of [node2vec](https://github.com/aditya-grover/node2vec). Implementation and optimization description could be found in this application note: [link to manuscript]()

## Requirements

* python==3.7.4
* gensim==3.8.0
* numpy==1.17.4
* numba==0.46.0

## Basic Usage

### Example

To run *node2vec* on Zachary's karate club network using `SparseOTF` mode, execute the following command from the project home directory:

```bash
python -m n2v --input demo/karate.edg --output demo/karate.emb --mode SparseOTF
```

### Demo

Execute the following command for full demonstration:

```bash
sh demo/run_n2v
```

### Mode

Three different modes are available, each of which is better optimized for different network sizes/densities

* `PreComp` (default) - precompute second order transition probabilities, using CSR graph
* `SparseOTF` - transition probabilites computed on-the-fly, using CSR graph
* `DenseOTF` - transition probabilities computed on-the-fly, using dense matrix

### Input

The supported input format is edgelist `.edg` file (node id could be int or string):

```
node1_id node2_id <weight_float, optional>
```

Another supported input format (only for DenseOTF) is numpy array `.npz` file, run the following command to prepare the `.npz` file from `.edg`

```bash
python -m n2v --input $input_edgelist --output $output_npz --task todense
```

### Output

The output files has *n+1 lines for graph with *n* vertices, with a header line of the following format:

```
num_of_nodes dim_of_representation
```

The following  next *n* lines are the representations of dimension $d$ following the corresponding node ID:

```
node_id dim_1 dim_2 ... dim_d
```
