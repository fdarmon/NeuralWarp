# NeuralWarp: Improving neural implicit surfaces geometry with patch warping

## [Project page](http://imagine.enpc.fr/~darmonf/NeuralWarp/) | [Paper](https://arxiv.com)
Code release of paper Improving neural implicit surfaces geometry with patch warping\
[François Darmon](http://imagine.enpc.fr/~darmonf), Bénédicte Bascle, Jean-Clément Devaux, [Pascal Monasse](https://imagine.enpc.fr:/monasse) and [Mathieu Aubry](http://imagine.enpc.fr/~aubrym/)

## Installation

See `requirements.txt` for the python packages. 


## Data

Download data with
`./download_dtu.sh` and `./download_epfl.sh`

## Extract mesh from a pretrained model

Download the pretrained models with `./download_pretrained_models.sh` then 
run the extraction script

`python extract_mesh.py --conf confs/NeuralWarp.conf --dataset DATASET --scene SCENE`
- `DATASET` can be either `dtu`or `epfl`
- `SCENE` is the scan id for DTU data and either `fountain` or `herzjesu` for EPFL.
  
The output mesh will be in `evals/NeuralWarp_SCENE/ouptut_mesh.ply`
be 

You can also run the evaluation: first download DTU evaluation data `./download_dtu_eval`,
then run the evaluation script `python eval.py --dataset DATASET --scene SCENE`. It will write the evaluation metrics in `evals/NeuralWarp_SCENE/result.txt`

## Train a model from scratch

First train a baseline model (i.e. VolSDF)
`python train.py --conf confs/baseline_DATASET`

Then finetune using our method with `python train.py --conf confs/NeuralWarp_DATASET`