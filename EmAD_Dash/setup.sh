#!/bin/bash

echo "This script will install EmAD requirements and set its python environment as the systems default."

# 1- Download Berryconda3, make it executable, Install it, and delete it
wget https://github.com/jjhelmus/berryconda/releases/download/v2.0.0/Berryconda3-2.0.0-Linux-armv7l.sh
chmod +x Berryconda3-2.0.0-Linux-armv7l.sh
./Berryconda3-2.0.0-Linux-armv7l.sh -b
rm Berryconda3-2.0.0-Linux-armv7l.sh

# 2- Export berryconda path, this will make the conda installation the defualt
echo "export PATH=$HOME/berryconda3/bin:$PATH" >> $HOME/.profile
export PATH=$HOME/berryconda3/bin:$PATH

# 3- Update conda
conda update -y conda

# 4- Update pip from conda v9.x -> v18.x
conda update -y pip

# 5- Update pip from pip v18.x -> v20.x
pip install --upgrade  pip

# 6- Install needed packages from conda
conda install -y -c numba numba
conda install -y scikit-learn matplotlib pandas

# 7- Install pyod and nose from piwheels for fast installation
pip install pyod nose -i https://www.piwheels.org/simple

# 8- Finish
echo "For changes to apply, please reboot or run the command: export PATH=$HOME/berryconda3/bin:$PATH"

