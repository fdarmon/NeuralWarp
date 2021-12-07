import argparse
from evaluation import dtu_eval, epfl_eval
from pathlib import Path
from pyhocon import ConfigFactory


parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str, required=True)
parser.add_argument("--suffix", default="")
parser.add_argument("--confname")

args = parser.parse_args()
scene = args.scene
expname = Path(args.confname).with_suffix("").name

evaldir = Path(f"evals/{expname}_{scene}")
inp_mesh_path = evaldir / f"output_mesh{args.suffix}.ply"

conf = ConfigFactory.parse_file(args.confname)
if conf["dataset"]["data_dir"] == "DTU":
    dtu_eval.eval(inp_mesh_path, int(scene), "data/dtu_eval", evaldir, args.suffix)
else:
    epfl_eval.eval(inp_mesh_path, scene, "data/epfl", evaldir, args.suffix)