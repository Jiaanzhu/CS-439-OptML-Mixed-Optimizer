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

- section-4a: 
	- run.py: scripts to obtain results for optimizer with Adam and SGD applied on different layers
	- figures: figures plotted by run.py and found in report.pdf
- section-4b: 
	- MAS_variants.ipynb: notebook with Mixed Adam and SGD applied on all layers
	- figures: figures plotted in MAS_variants.ipynb and found in report.pdf

- requirements.txt: library needed
- report.pdf: the report in which we explain and share our results

### Run

To obtain 'mixing of optimizers in different layers' plots
```
cd section-4a
python3 run.py
```  

To obtain 'mixing of optimizers within same layer' plots
```
cd section-4b
jupyter notebook MAS_variants.ipynb
```

### Data

- MNIST
- CIFAR10
