# create a new conda env:

```bash
conda create -n python310cv python=3.10
```

# Activate your new Python 3.10 environment:

```bash
conda activate python310cv
```

# install ipykernel when logged in the new env:

```bash
conda install ipykernel pandas numpy seaborn scikit-learn
```

# install the ipython kernel at user level

```bash
ipython kernel install --user --name=python310cv
```

# conda install requirements

```bash
conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
