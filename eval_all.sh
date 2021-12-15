for scene in 105 106 110 114 118 122 24 37 40 55 63 65 69 83 97
do
#  python extract_mesh.py --conf confs/NeuralWarp.conf --scene $scene --dataset dtu --dilation_radius 25 --suffix _25
#  python eval.py --conf confs/NeuralWarp.conf --scene $scene --dataset dtu --suffix _25
#
#  python extract_mesh.py --conf confs/NeuralWarp.conf --scene $scene --dataset dtu --dilation_radius 50 --suffix _50
#  python eval.py --conf confs/NeuralWarp.conf --scene $scene --dataset dtu --suffix _50
  for conf in PatchNoOcc PatchNoVol Pixel VolSDF
  do
    python extract_mesh.py --conf confs/ablations/$conf.conf --scene $scene --dataset dtu --dilation_radius 12 --suffix _12
    python eval.py --conf confs/ablations/$conf.conf --scene $scene --dataset dtu --suffix _12
  done
done

