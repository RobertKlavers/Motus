apt-get update
apt-get upgrade

apt-get install -y git pkg-config
apt-get install -y python-pip python-dev build-essential
apt-get install -y python-numpy python-matplotlib

export LC_ALL=C
pip install -U pip
pip install --upgrade setuptools
pip install scikit-image
pip install -r /vagrant/requirements.txt

pip install opencv-python