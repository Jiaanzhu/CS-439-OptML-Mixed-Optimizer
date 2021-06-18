# OptML Project
In this project we try multiple ways of combining SGD and Adam in order to obtain properties of both.

### Install

This project requires **Python** and the following Python libraries installed:

- [torch](https://pytorch.org)
- [matplotlib](http://matplotlib.org/) 
- torchvision
- tqdm
- pytorch_model_summary
- jupyter


### Code

- MAS_variants.ipynb: notebook with Mixed Adam and SGD applied on all layers
- Test_SGD.ipynb: notebook with Adam and SGD applied on different layers
- conv.py: 
- models.py
- trainer.py
- datasets.py
- requirements.txt

### Run

In a terminal or command window, navigate to the top-level project directory `ee559DL_project1` (that contains this README) and run one of the following commands:
```
jupyter notebook Test_SGD.ipynb
jupyter notebook MAS_variants.ipynb
```  


### Data

- MNIST
- CIFAR10
