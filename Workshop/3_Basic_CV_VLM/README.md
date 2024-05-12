# Setup Environment on Vertex Notebooks with terminal
1. create a new conda env:

```bash
conda create -n python310cv python=3.10
```

2. Activate your new Python 3.10 environment:

```bash
conda activate python310cv
```

3. install ipykernel when logged in the new env:

```bash
conda install ipykernel pandas numpy seaborn scikit-learn pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

4. install the ipython kernel at user level

```bash
ipython kernel install --user --name=python310cv
```

5.  install requirements

```bash
pip install -r requirements.txt
```

6. Select the python310cv kernel for the notebookos

# Links for Images to test on/notebooks to load 
- dog1 :"https://i.ibb.co/Z6L3q4j/dog1.jpg"
- background: "https://i.ibb.co/NVwgY75/background.png"
- cv-logo: "https://i.ibb.co/yP4gRy7/cv-logo.png"
- hot-air-balloon: "https://i.ibb.co/RBgthgt/hot-air-balloon.jpg"
- beach-chair: "https://i.ibb.co/prG3Ctt/beach-chair.jpg"
- horserider: "https://i.ibb.co/r4Zfg15/horserider.jpg"
- motocycle-with-cars: "https://i.ibb.co/6YgXdfG/motocycle-with-cars.png"
- pedestrian: https://i.ibb.co/N64S5GP/Penn-Ped00048.png
