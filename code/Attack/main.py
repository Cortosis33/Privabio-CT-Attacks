#!/usr/bin/env python3.7

"""!
File main
Use for make the call of attack by quadratic solver with Gurobi.
"""
from modele_322 import *

global path, size_template, size_image

## path to retrive the database of the digital print.
path = '../../BDD/image/DB1_B/'
## The size of the template use. Value in [1,∞].
size_template = 64
## The size of the image use. If value is -1 the image is complete else the image is the middle of the original digital print with a size depending on this value.
size_image = 5
start = time()

# images = create_all_images(path, size_template, size_image, size_image)

# all_322(images)
# all_322(images, True)

# attacker = create_image(path, "101_1.tif", size_template, size_image, size_image)
# target = create_image(path, "105_6.tif", size_template, size_image, size_image)
# test_322(attacker, target, objective= False)

# max_filtered_value_model()




print("\n Time : ",time()-start)
