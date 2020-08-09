#!/bin/bash

# 1- Make the working directory
mkdir emad;cd emad

# 2- update packages and install the needed apps
apt update
apt install -y wget bzip2

# 3- Download Berryconda3, make it executable, and Install it
wget https://github.com/jjhelmus/berryconda/releases/download/v2.0.0/Berryconda3-2.0.0-Linux-armv7l.sh
chmod +x Berryconda3-2.0.0-Linux-armv7l.sh
./Berryconda3-2.0.0-Linux-armv7l.sh -b

# 4- Export berryconda path
export PATH="/root/berryconda3/bin:$PATH"

# 5- Update conda
conda update conda
conda info

# 6- Update pip from conda v9.x -> v18.x
conda update -y pip

# 7- Update pip from pip v18.x -> v20.x
pip install --upgrade  pip

# 8- Install needed packages from conda
conda install -y -c numba numba
conda install -y scikit-learn matplotlib pandas

# 9- Install pyod from piwheels for fast installation
pip install pyod -i https://www.piwheels.org/simple

# 10- Start Jupyterlab for development
# jupyter-lab --ip=* --allow-root --no-browser --port 9999

# ADD RASPI Repository
nano /etc/apt/sources.list

deb http://archive.raspbian.org/raspbian buster main contrib non-free
deb-src http://archive.raspbian.org/raspbian buster main contrib non-free

wget https://archive.raspbian.org/raspbian.public.key -O - | sudo apt-key add -apt
