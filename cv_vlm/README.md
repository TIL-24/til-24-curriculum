# create a new conda env:
$ conda create -n python310cv python=3.10

# Activate your new Python 3.8 environment:
$ conda activate python310

#install ipykernel when logged in the new env:
(Python)$ conda install ipykernel pandas numpy seaborn scikit-learn

# install the ipython kernel at user level
(Python)$ ipython kernel install --user --name=python38

# conda install requirements
conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt