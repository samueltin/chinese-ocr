conda create -n chinese-ocr python=2.7 pip scipy numpy PIL jupyter##运用conda 创建python环境
source activate chinese-ocr
pip install easydict
pip install keras==2.0.8
pip install Cython opencv-python
pip install matplotlib
pip install -U pillow
pip install  h5py lmdb mahotas
#conda install pytorch=0.1.12 torchvision -c soumith
#conda install tensorflow=1.3 ##解决cuda报错相关问题
pip3 install torch torchvision
pip install tensorflow==1.3
cd ./ctpn/lib/utils
sh make-for-cpu.sh


