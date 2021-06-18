# OptML Project
We wish to try multiple ways of combining SGD and Adam in order to obtain properties of both.

### Install

This project requires **Python** and the following Python libraries installed:

- [torch](https://pytorch.org)
- [matplotlib](http://matplotlib.org/) 
- torchvision
- tqdm
- pytorch_model_summary
- jupyter


### Code

In this github you will find:

   - src: This is a file that contain all the needed functions and class for the MLP that will be tested in test.py and also in particular:
        - models.py : This class contains the 2 CNN (single and double channels) and the MLP with an argument to add or not an auxiliary loss
        - train.py : This function train the architecture wanted
        - utils.py: Contain useful functions
   - data: This file contains the MNIST dataset
   - media: Store images useful for report or Readme.md
   - test.py: Python code that run the test 
   - dlc-miniprojects.pdf : This is the subject of this project 1
   - requirements.txt : This a text file that can be used to download the library presented before 
   - demo.ipynb : This jupyter notebook show how we train the MLP and the two CNN 
   - experiment.ipynb : This jupyter notebook contains all the plots. This notebook aims to give a sense to those results by giving relevant and explicit plots that show the accuracy but also the convergence of our achitectures (this is where we use plotly and matplotly)

### Run

In a terminal or command window, navigate to the top-level project directory `ee559DL_project1` (that contains this README) and run one of the following commands:
```
jupyter notebook Test_SGD.ipynb
jupyter notebook MAS_variants.ipynb
```  


### Data

- MNIST
- CIFAR10
