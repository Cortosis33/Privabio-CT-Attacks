
"""!
File EER
Contain functions use to calculate EER with the distance between template.
"""
from image import *

def make_folder(attacker, last_folder):
    """!
    make_folder function, make a list of folders for save data with write function.
    The path is /EER_template/bdd_***/template_***/image_***x***/FAR/.

    @param attacker Image object of the attacker. Use for the size of the image.
    @param last_folder is a string and the last folder add to the list. Actually use for add FRR or FAR folder

    @return the list of folder, ["EER_template","bdd_***","template_***","image_***x***","FAR"]
    """
    folder = ["EER_template"]
    folder.append("bdd_"+attacker.bdd)
    folder.append("template_"+str(attacker.size_template))

    size_image = str(attacker.n)+"x"+str(attacker.m)
    folder.append("image_"+size_image)
    folder.append(last_folder)
    # if target!=None:
    #     attack = str(attacker.number_image) + "->" + str(target.number_image)
    #     folder.append("attack_"+attack)
    return folder

def FRR(images):
    """!
    FRR function, calculate all the distance template for all couple of image with the same user.

    @param images list Image object. Normally use all the image of the database.

    @return the list of distance templace calculate.
    """
    tab = []
    limit = len(images)
    # limit = 1
    for i in range(limit):
        print(" FRR",i)
        min_for = min(i+1, limit)
        max_for = min(min_for + 8, limit)
        attacker = images[i]
        for j in range(min_for, max_for):
            # print(" FRR",i,j)
            target = images[j]
            if (j!=i and attacker.user == target.user):
                distance = distance_template_norm(attacker.template, target.template)
                tab.append(distance)
    return tab

def FAR(images):
    """!
    FAR function, calculate all the distance template for all couple of image with a different user.

    @param images list Image object. Normally use all the image of the database.

    @return the list of distance templace calculate.
    """
    tab = []
    limit = len(images)
    # limit = 1
    for i in range(limit):
        print(" FAR",i)
        # min = int(i/8+1)*8
        # max = len(images)
        min_for = min(i+1, limit)
        max_for = min(min_for + 8, limit)
        # max = 80
        attacker = images[i]
        old_attacker_matrix = attacker.matrix
        old_target_user = -1
        for j in range(min_for, max_for):
            # print(" FAR",i,j)
            target = images[j]
            if (j!=i and attacker.user != target.user):# and i%8 ==j%8):
                if old_target_user != target.user:
                    old_target_user = target.user
                    if target.matrix == attacker.matrix:
                        attacker.change_matrix(images[j+1].matrix)
                    else:
                        attacker.change_matrix(target.matrix)
                # TODO
                # attacker_template = template_no_class(attacker.image, target.matrix)
                distance = distance_template_norm(attacker.template, target.template)
                tab.append(distance)
        attacker.change_matrix(old_attacker_matrix)
    return tab

def EER(FRR, FAR, number_threshold, images):
    """!
    EER function, calculate the EER and the threshold with the FAR and FRR gives and show the graph

    @param FRR list Image object. Normally use all the image of the database.
    @param FAR list Image object. Normally use all the image of the database.
    @param number_threshold the number of threshold calculate. Greater he is more the result is accurate.
    @param images list Image object. Normally use all the image of the database.

    @return the distance limit to accept user autentification.
    """
    frr, far = calcul_EER_lines(FRR, FAR, number_threshold)
    min_distance = 100
    index = 0
    for threshold in range(number_threshold):
        if abs(far[threshold] - frr[threshold]) < min_distance:
            min_distance = abs(far[threshold] - frr[threshold])
            index = threshold

    print_EER(index, number_threshold, frr, far)
    eer = round((far[index] + frr[index])/2,3)
    threshold = round(index/number_threshold*100,3)
    title = " Threshold : " + str(threshold)
    title += ", EER : " + str(eer)
    title += ", size : " + str(images[0].n) + "x" + str(images[0].m)
    title += ", size template : " + str(images[0].size_template)
    title += ", BDD : " + str(images[0].bdd)
    show_EER(frr, far, number_threshold, title, eer, threshold)
    return index/number_threshold*100

def EER_file(file_FRR, file_FAR, number_threshold):
    """!
    EER_file function, call  EER function with the data in the files gives in parameter. Not use now

    @param file_FRR file path for find data of FRR.
    @param file_FAR file path for find data of FAR.
    @param number_threshold the number of threshold calculate. Greater he is more the result is accurate.

    @return the distance limit to accept user autentification.
    """
    FRR = retrieve_first_line_from_file(file_FRR)
    FAR = retrieve_first_line_from_file(file_FAR)
    print(file_FRR, len(FRR))
    print(file_FAR, len(FAR))
    if FRR == 0 or FAR == 0:
        print(" Bad Name for files")
        return
    threshold = EER(FRR, FAR, number_threshold)
    return threshold

def EER_start(images, number_threshold):
    """!
    EER_start function, call FRR, FAR and EER function.

    @param images list Image object. Normally use all the image of the database.
    @param number_threshold the number of threshold calculate. Greater he is more the result is accurate but up the time execution too.

    @return the distance limit to accept user authentification.
    """
    frr = FRR(images)
    far = FAR(images)
    # size_image = str(images[0].n)+"x"+str(images[0].m)
    folder_FRR = make_folder(images[0], "FRR")
    folder_FAR = make_folder(images[0], "FAR")
    # (write(frr, folder_FRR))
    # (write(far, folder_FAR))
    print(len(frr))
    print(len(far))
    threshold = EER(frr, far, number_threshold, images)
    return threshold

global path, size_template, size_image

# path = '../BDD/image/bmp/'
path = '../BDD/image/DB1_B/'
# path = '../BDD/image/DB2_B/'
start = time()
size_template = 64
size_image = -1

number_threshold = 1000

images = create_all_images(path, size_template, size_image, size_image)

EER_start(images, number_threshold)





print(time() - start)
