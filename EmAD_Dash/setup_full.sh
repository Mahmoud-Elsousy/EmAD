#!/bin/bash

# ref: https://askubuntu.com/a/30157/8698
if ! [ $(id -u) = 0 ]; then
   echo "Please run the script with as root." >&2
   exit 1
fi

if [ $SUDO_USER ]; then
    real_user=$SUDO_USER
else
    real_user=$(whoami)
fi

# 1- Make the working directory
sudo -u $real_user mkdir emad;cd emad

# 2- update packages and install the needed apps
apt update
apt install -y wget bzip2

# 3- Download Berryconda3, make it executable, Install it, and delete it
sudo -u $real_user wget https://github.com/jjhelmus/berryconda/releases/download/v2.0.0/Berryconda3-2.0.0-Linux-armv7l.sh
sudo -u $real_user chmod +x Berryconda3-2.0.0-Linux-armv7l.sh
sudo -u $real_user ./Berryconda3-2.0.0-Linux-armv7l.sh -b
sudo -u $real_user rm Berryconda3-2.0.0-Linux-armv7l.sh

# 4- Export berryconda path, this will make the conda installation the defualt
sudo -u $real_user echo "export PATH=$HOME/berryconda3/bin:$PATH" >> $HOME/.profile

# 5- Update conda
sudo -u $real_user conda update -y conda

# 6- Update pip from conda v9.x -> v18.x
sudo -u $real_user conda update -y pip

# 7- Update pip from pip v18.x -> v20.x
sudo -u $real_user pip install --upgrade  pip

# 8- Install needed packages from conda
sudo -u $real_user conda install -y -c numba numba
sudo -u $real_user conda install -y scikit-learn matplotlib pandas

# 9- Install pyod from piwheels for fast installation
sudo -u $real_user pip install pyod -i https://www.piwheels.org/simple

# 10- Install nose from piwheels for fast installation
sudo -u $real_user pip install nose -i https://www.piwheels.org/simple

# 11- Clean after installation to save space
sudo -u $real_user conda clean -tipsy \
&& find /root/berryconda3/ -type f,l -name '*.a' -delete \
&& find /root/berryconda3/ -type f,l -name '*.pyc' -delete \
&& find /root/berryconda3/ -type f,l -name '*.js.map' -delete \
&& rm -rf /root/berryconda3/pkgs

apt-get clean && rm -rf /var/lib/apt/lists/*
