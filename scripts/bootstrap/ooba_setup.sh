set -e
mkdir -p ~/ooba-hybrid/webui
cd ~/ooba-hybrid/webui
python3 -m venv venv --system-site-packages
echo 'source ~/ooba-hybrid/webui/venv/bin/activate' >> ~/.bashrc
source venv/bin/activate
pip install --upgrade pip wheel setuptools