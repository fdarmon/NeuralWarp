cd data

wget http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip
wget http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip

unzip SampleSet.zip

mv "SampleSet/MVS\ Data/" dtu_eval
rm -r SampleSet

cd dtu_eval
unzip Points.zip