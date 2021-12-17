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

`python extract_mesh.py --conf CONF --scene SCENE [--OPTIONS]`
- `CONF` is the configuration file (e.g. `confs/NeuralWarp_dtu.conf`)
- `SCENE` is the scan id for DTU data and either `fountain` or `herzjesu` for EPFL.
- See `python extract_mesh.py --help` for a detailed explanation of the options. 
  The evaluation in the papers are with default options for DTU and with `--bbox_size 4 --no_one_cc --filter_visible_triangles --min_nb_visible 1` for EPFL.
  
The output mesh will be in `evals/CONF_SCENE/ouptut_mesh.ply`

You can also run the evaluation: first download the DTU evaluation data `./download_dtu_eval`,
then run the evaluation script `python eval.py --scene SCENE`.
The evaluation metrics will be written in `evals/CONF_SCENE/result.txt`.

## Train a model from scratch

First train a baseline model (i.e. VolSDF)
`python train.py --conf confs/baseline_DATASET --scene SCENE`.

Then finetune using our method with `python train.py --conf confs/NeuralWarp_DATASET --scene SCENE`.

A visualization html file is generated for each training in `exps/CONF_SCENE/TIMESTAMP/visu.html`.