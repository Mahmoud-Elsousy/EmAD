#!/bin/bash

# 1- Make the working directory
mkdir emad;cd emad

# 2- update packages and install the needed apps
apt update
apt install -y wget bzip2

# 3- Download Berryconda3, make it executable, Install it, and delete it
wget https://github.com/jjhelmus/berryconda/releases/download/v2.0.0/Berryconda3-2.0.0-Linux-armv7l.sh
chmod +x Berryconda3-2.0.0-Linux-armv7l.sh
./Berryconda3-2.0.0-Linux-armv7l.sh -b
rm Berryconda3-2.0.0-Linux-armv7l.sh

# 4- Export berryconda path, this will make the conda installation the defualt
export PATH="$HOME/berryconda3/bin:$PATH"

# 5- Update conda
conda update -y conda

# 6- Update pip from conda v9.x -> v18.x
conda update -y pip

# 7- Update pip from pip v18.x -> v20.x
pip install --upgrade  pip

# 8- Install needed packages from conda
conda install -y -c numba numba
conda install -y scikit-learn matplotlib pandas

# 9- Install pyod from piwheels for fast installation
pip install pyod -i https://www.piwheels.org/simple

# 10- Install nose from piwheels for fast installation
pip install nose -i https://www.piwheels.org/simple

# 11- Clean after installation to save space
conda clean -tipsy \
&& find /root/berryconda3/ -type f,l -name '*.a' -delete \
&& find /root/berryconda3/ -type f,l -name '*.pyc' -delete \
&& find /root/berryconda3/ -type f,l -name '*.js.map' -delete \
&& rm -rf /root/berryconda3/pkgs

apt-get clean && rm -rf /var/lib/apt/lists/*
