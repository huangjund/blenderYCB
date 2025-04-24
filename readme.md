# blenderYCB 
readme will come soon...

This is the official source code for the high fidelity rendering of both single objects or chained robots. This realization is part of the paper: HandCept: A Visual-Inertial Fusion Framework for Robust Proprioception in Dexterous Hands.
generate URDF from solidworks

```angular2html
mkdir output
mkdir data
mkdir data_syn

# change configs files before below steps
cd data
ln -s /ssd2/T-DexCO\ Hand/objs_custom_data/ . # link the obj models to the data folder
ln -s /ssd2/T-DexCO\ Hand/AssemV_camera/ . # link the urdf folder to the data folder
ln -s /ssd2/sun2012pascalformat/ . # link the background images to the data folder

# create a conda environment
conda create -n render python=3.10 -y
pip -r install requirement.txt
```

```angular2html
# edit config.json

# generate data
blenderproc run scripts/generateRendering.py
python vis_hdf5.py
python hdf5Topng.py
```
## for YCB data generation, the valid id should start from 1