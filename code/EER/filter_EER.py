
"""!
File filter_EER
Contain functions use to calculate EER with the distance between feature.
Almost all the function have the same functioning than the funcion in EER file.
"""
from image import *

# make the list of the folder for save result of EER
def make_folder(attacker, last_folder):
    folder = ["EER_filter"]
    folder.append("bdd_"+attacker.bdd)
    folder.append("template_"+str(attacker.size_template))

    size_image = str(attacker.n)+"x"+str(attacker.m)
    folder.append("image_"+size_image)
    folder.append(last_folder)
    # if target!=None:
    #     attack = str(attacker.number_image) + "->" + str(target.number_image)
    #     folder.append("attack_"+attack)
    return folder
start = time()

# return all the distance feature between all image with the same user
def FRR(images):
    tab = []
    limit = len(images)
    # limit = 1
    for i in range(limit):
        print("FRR",i)
        min_for = min(i+1, limit)
        max_for = min(min_for + 8, limit)
        attacker = images[i]
        for j in range(min_for, max_for):
            target = images[j]
            if (j!=i and attacker.user == target.user):
                distance = distance_feature_norm(attacker.feature, target.feature, target.m)
                tab.append(distance)
    return tab

# return all the distance feature between all image with a different user
def FAR(images):
    tab = []
    limit = len(images)
    # limit = 1
    for i in range(limit):
        print("FAR",i)
        min_for = min(i+1, limit)
        max_for = min(min_for + 8, limit)
        # max = 80
        attacker = images[i]
        for j in range(min_for, max_for):
            target = images[j]
            if (j!=i and attacker.user != target.user):
                distance = distance_feature_norm(attacker.feature, target.feature, target.m)
                tab.append(distance)
    return tab

# with the distance calculate in FRR and FAR calcul the EER
def EER(FRR, FAR, max_threshold):
    frr, far = calcul_EER_lines(FRR, FAR, max_threshold)
    # show_EER(frr, far, max_threshold)
    min_distance = 100
    index = 0
    for threshold in range(max_threshold):
        if abs(far[threshold] - frr[threshold]) < min_distance:
            min_distance = abs(far[threshold] - frr[threshold])
            index = threshold
    print_EER(index, max_threshold, frr, far)
    title = " Threshold : " + str(round(index/max_threshold*100,3))
    title += ", EER : " + str(round((far[index] + frr[index])/2,3))
    title += ", size : " + str(images[0].n) + "x" + str(images[0].m)
    title += ", BDD : " + str(images[0].bdd)
    show_EER(frr, far, max_threshold, title)
    return index/max_threshold*100
    return (frr, far)

# calcul the EER with the result of FRR and FAR in file
def EER_file(file_FRR, file_FAR, max_threshold):
    FRR = retrieve_first_line_from_file(file_FRR)
    FAR = retrieve_first_line_from_file(file_FAR)
    print(file_FRR, len(FRR))
    print(file_FAR, len(FAR))
    if FRR == 0 or FAR == 0:
        print(" Bad Name for files")
        return
    eer = EER(FRR, FAR, max_threshold)
    return eer

# Calcul FAR, FRR after that EER and save the result of FRR and FAR
def EER_start(images, max_threshold):
    frr = FRR(images)
    far = FAR(images)
    size_image = str(images[0].n)+"x"+str(images[0].m)
    folder_FRR = make_folder(images[0], "FRR")
    folder_FAR = make_folder(images[0], "FAR")
    # write(frr, folder_FRR, "_test")
    # write(far, folder_FAR, "_test")
    print(len(frr))
    print(len(far))
    eer = EER(frr, far, max_threshold)
    return eer



global path, BDD, size_template, size_image

path = '../BDD/image/DB1_B/'
BDD = path.split("/")[-2]



# images = modify_all_images(images,20, 20)
# show_image(images[0])
size_template = 256
size_image = -1
images_bdd = create_all_images(path, 64, 10, 10)
# EER_start(images, 1000)
print("la")
images = [0]*80
max_threshold = 1000
# for i in [64,128,256,512,1024,2048]:
#     size_template = i
#     # for j in range(80):
#     #     im = random_image(374,388)
#     #     images[j] = Image(im, size_template, BDD, images_bdd[j].name_image)
#     images = create_all_images(path, size_template, size_image, size_image)
#     EER_start(images, max_threshold)
# size_template = 256
# for i in [10,100,200,300,-1]:
#     size_image = i
#     for j in range(80):
#         im = random_image(size_image,size_image)
#         images[j] = Image(im, size_template, BDD, images_bdd[j].name_image)
#     # images = create_all_images(path, size_template, size_image, size_image)
#     EER_start(images, max_threshold)

# path_file = "../result/EER_filter/DB1_B/512/20x20/"
# file_FAR = path_file+"FAR/0.csv"
# file_FRR = path_file+"FRR/0.csv"
# eer = EER_file(file_FRR, file_FAR, max_threshold)
# print (eer)

# 10,100,200,300
full_256 = [51.177,48.157,55.698,15.031,44.877]
first_256 = [51.749,43.346,41.94,38.866,39.163]
second_256 = [51.732,50.793,46.36,36.583,37.551]
# random_256 = [52.509,47.923,46.508,53.459,53.272]
_256 = [full_256, first_256, second_256]
_256_name = ["original", "first delete", "second delete"]

show_plot(_256, _256_name, "Variation of distance feature with differents size of image and different BDD", "size image (10,100,200,300,374*388)", "Distance Feature")
#
file = "0_test.csv"

number = split('_| ', file)

print(number)





print(time() - start)
