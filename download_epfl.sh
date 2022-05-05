mkdir -p data

cd data

echo "Downloading EPFL GT ..."
wget http://imagine.enpc.fr/~darmonf/NeuralWarp/epfl.zip
unzip epfl.zip

cd epfl

echo "Downloading the EPFL images ..."
wget https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/data/fountain_dense/urd/fountain_dense_images.tar.gz
wget https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/data/herzjesu_dense/urd/herzjesu_dense_images.tar.gz

echo "Extracting ..."
tar -zxvf fountain_dense_images.tar.gz
tar -zxvf herzjesu_dense_images.tar.gz

