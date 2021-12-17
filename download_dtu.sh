mkdir -p data
cd data
echo "Downloading the DTU dataset ..."
wget https://www.dropbox.com/s/ujmakiaiekdl6sh/DTU.zip
wget http://imagine.enpc.fr/~darmonf/NeuralWarp/dtu_supp.zip
echo "Start unzipping ..."
unzip DTU.zip
unzip dtu_supp.zip
echo "DTU dataset is ready!"