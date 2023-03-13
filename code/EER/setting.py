
"""!
File setting
Contain the import and the global variable of all the project.
"""
import copy
import imageio
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import seaborn as sns
import random
import scipy.ndimage
from math import *
from statistics import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cv2
import os
import imageio
from time import *
import csv
from pathlib import Path
from re import *
from turtle import title

## The seed selected to keep the repeatability of the algorithm.
selected_seed = "doferreira1"
random.seed(selected_seed)

## path to retrive the database of the digital print.
# path = '../../BDD/image/bmp/'
path = '../../BDD/image/DB1_B/'
## The size of the template use. Value in [1,∞].
size_template = 256
## The size of the image use. If value is -1 the image is complete else the image is the middle of the original digital print with a size depending on this value.
size_image = 10
## The number use in suffixe of the different file name. If the value is -1 the function write gonna take a number avaible to keep all the existant files.
## The fucntion multiple optimisation gonne retrieve the data file if it exist if this parameter is different than 1.
## So if you want create new file of data you have to take a new suffix never use or -1 but -1 is inapporpiate for save data files we can't retrieve the file after that.
number_file = -1
## Boolean value. If true the image gonna be save in a folder.
want_write_image = False
## Threshold use in the evaluation. When the distance template is smaller than threshold the objective value gonne take a distance template at 0. Value in [0,1].
threshold_evaluation = 0
## The limit of time in the resolution of the quadratic model with gurobi in seconds.
time_gurobi = 10

## Weight of the image in the objective function. Value in [0,1]
weightI = 0.1

## Dictionary use to retrieve the string for create a folder to save data depending on the extention of file use. Use in write function.
switch_folder = {".csv" : ["data"],
".bmp" : ["image"],
".png" : ["plot"]}

# genetique parameter
## The number of iteration whiout any improve in genetic algorithm accepted before stop the execution. Value in [1,∞].
max_while = 4
## The number of person selected in the genetic algorithm. Value in [1,∞].
number_select = 12
## The probability of a pixel mutate with a new value. Value in [0,1].
proba_mutation = 0.03
## Boolean value. If true the image in genese of the genetic algorithm gonna be random, else it's a copy of the attacker image.
random_genese = False

# gradiant parameter
## The maximum value accepted for the new value of the pixel in the change_pixel of the gradiant algorithm. Value in [1,255].
delta = 10
## Boolean value. If true the gradiant algorithm gonna use the function change_pixel else use the function change_pixel2.
change1 = True
