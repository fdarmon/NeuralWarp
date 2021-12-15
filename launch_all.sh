for scene in 105 106 110 114 118 122 24 37 40 55 63 65 69 83 97
do
  args="--conf confs/ablations/NeuralWarp.conf --scene $scene" sbatch ./jz_train.sh
  args="--conf confs/ablations/PatchNoOcc.conf --scene $scene" sbatch ./jz_train.sh
  args="--conf confs/ablations/PatchNoVol.conf --scene $scene" sbatch ./jz_train.sh
done

