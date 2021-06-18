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
- typing


### Code

- MAS_variants.ipynb: notebook with Mixed Adam and SGD applied on all layers
- run.py: scripts to obtain results for optimizer with Adam and SGD applied on different layers
- requirements.txt: library needed
- report.pdf: the report in which we explain and share our results
- figures: folder with figures plotted, and appearing in the report

### Run

In a terminal or command window, navigate to the top-level project directory `ee559DL_project1` (that contains this README) and run one of the following commands:
```
python3 run.py
jupyter notebook MAS_variants.ipynb
```  


### Data

- MNIST
- CIFAR10
