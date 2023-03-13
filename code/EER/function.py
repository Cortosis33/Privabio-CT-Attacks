
from setting  import *
"""!
function file.
Contains Person class and the function use in all the project.
"""
class Person:
    """!
    Class Person.
    Contains attributes and methods needed by different attacks to keep all the necessaries data.
    The name of the class as to be changed. It was create with the genetic algorithm but now the name is outdated.
    """
    def __init__(self, image, matrix, old_image, weightI, target_template, distance_template=1, distance_image=1):
        """!
        The constructor of the class Person.

        @param image the image of digital print. it is the image that will be modifies in the attack.
        @param matrix the matrix of the target in the attack.
        @param old_image the orginal image of the attacker before modification.
        @param weightI the weight of the image in the objective function.
        @param target_template the template of the target.
        @param distance_template the distance template in the objective function. Default value 1.
        @param distance_image the distance image in the objective. Default value 1.
        """
        weightT = 1 - weightI
        ## Two dimension list with value in [0,255] for the value of the pixel. This is the modified image of the attacker in the attack.
        self.image = image
        ## Two dimension list with value in [-0.5,0.5]. This is the matrix of the target in the attack.
        self.matrix = matrix
        ## One dimension list with value in {0,1}. This is the template of the target in tha attack.
        self.target_template = target_template
        ## Two dimension list with value in [0,255] for the value of the pixel. This is the original image of the attacker before the attack.
        self.old_image = old_image
        ## The distance between the target_template and the new template of the attacker calculate with the new image. Value in [0,1].
        self.distance_template = distance_template
        ## The distance between the original image of the attacker and the modified image. Value in [0,1].
        self.distance_image = distance_image
        ## Value objective of the image calculate with the distance_image and the distance_template depending on the parameter weightI. Value in [0,1].
        self.objective = weightT * distance_template + weightI * distance_image

    def set_distance(self, distance_template, distance_image, weightI, threshold):
        """!
        Method for Person class. set_distance function, calculate the objective value with the two distance required and set the value obtain.
        If distance_template < threshold distance_template is considered to be at 0 for the objective value to stop degraded the image when the authentification is successful.

        @param distance_template the distance template in the objective function.
        @param distance_image the distance image in the objective.
        @param weightI the weight of the image in the objective function. Take real value between 0 and 1.
        @param threshold the threshold use by the system to accept authentification.
        """
        weightT = 1 - weightI
        self.distance_template = distance_template
        self.distance_image = distance_image
        if distance_template < threshold:
            self.objective = weightI * distance_image
        else:
            self.objective = weightT * distance_template + weightI * distance_image

    def evaluation(self, weightI, threshold):
        """!
        Method for Person class. evaluation function, calculate the objective value of attacker. Set the value in attacker.objective.

        @param weightI the weight of the image in the objective function. Take real value between 0 and 1.
        @param threshold the threshold use by the system to accept authentification. Take real value between 0 and 1.

        @return the value objective of the attacker
        """
        template = template_no_class(self.image, self.matrix)
        distance_template = distance_template_norm(template, self.target_template)
        distance_image = distance_image_norm(self.image, self.old_image)
        self.set_distance(distance_template, distance_image, weightI, threshold)
        return self.objective

def retrive_user(name_image):
    """!
    retrive_user function, retrieve the user number with the name of the file. Work with the database DB1_B and DB2_B.

    @param name_image the name of the digital print image.

    @return the number of user find in the name_image.
    """
    test = split('_| |\.', name_image)
    return int(test[0])

def random_image(n, m):
    """!
    random_image function, create a random image with value between 0 and 255.

    @param n the number of line in the image.
    @param m the number of column in the image.

    @return the image create.
    """
    image = [0]* n
    for i in range(n):
        temp = [0] * m
        for j in range(m):
            rand = random.randint(0,255)
            # rand = random.sample([0,255],1)[0]
            temp[j] = rand
        image[i] = temp[:]
    return image

def variation_image(attacker, delta):
    """!
    variation_image function, make a variation in all pixel of the image. Use for some test in genese in genetic algorithm. Not usefull.

    @param attacker Image object of attacker.
    @param delta the gap value max accepted.

    @return the image create.
    """
    n = len(attacker)
    m = len(attacker[0])
    image = [0] * n
    for i in range(n):
        temp = [0] * m
        for j in range(m):
            rand = random.randint(max(0,attacker[i][j]-delta),min(255,attacker[i][j]+delta))
            temp[j] = rand
        image[i] = temp[:]
    return image

def blurry_image(attacker):
    """!
    blurry_image function, make a mean of the four pixel around each pixel to make a new value. Use for some test in genese in genetic algorithm. Not usefull.

    @param attacker Image object of attacker.

    @return the image create.
    """
    n = len(attacker)
    m = len(attacker[0])
    image = [0] * n
    for i in range(n):
        temp = [0] * m
        for j in range(m):
            middle = int(attacker[i][j])
            if i != 0 and j != 0 and i != n-1 and j != m-1:
                first = int(attacker[i-1][j])
                second = int(attacker[i][j-1])
                third = int(attacker[i][j+1])
                fourth = int(attacker[i+1][j])
                new_pixel = int((first + second + third + fourth+ middle)/5)
                temp[j] = new_pixel
            else:
                temp[j] = middle
        image[i] = temp[:]
    return image

def print_tab(tab):
    """!
    print_tab function, print each row of a list.

    @param tab the list to print.
    """
    for row in tab:
        print(row)

def all_images_name(path):
    """!
    all_images_name function, create a list with all names files in a folder.

    @param path, string value of the path of the folder. Exemple '../../BDD/image/DB1_B/'

    @return the list of names files.
    """
    images_name = []
    for filename in sorted(os.listdir(path)):
        images_name.append(filename)
    return images_name

def all_images(path):
    """!
    all_images function, create a list with all images find in a folder.

    @param path, string value of the path of the folder. Exemple '../../BDD/image/DB1_B/'

    @return the list of images create.
    """
    images = []
    images_name = all_images_name(path)
    for image_name in images_name:

        image = imageio.v2.imread(path+image_name)
        images.append(image)
    return images

def search_folder(folder):
    """!
    search_folder function, retrieve in which  parents folders the folder given is. Use for retrive the folder Result.

    @param folder the folder we are looking for.

    @return the path find. Exemple of result "./../../Result/"
    """
    path = "./"
    Exist = os.path.exists(path+folder)
    while not Exist:
        path += "../"
        Exist = os.path.exists(path+folder)
    return path

def create_path(root, new_folders):
    """!
    create_path function, create a path with the root and the list of folders given and create the folder if isn't exist.

    @param root string value of the root of the path. Exemple "result"
    @param new_folders the list of folder to add to the root. Exemple ["genetic","bdd_DB1_B","template_64","image_374x388","attack_***->***","target_matrix"]

    @return the path create. Exemple "./../../result/genetic/bdd_DB1_B/template_64/image_374x388/attack_***->***/target_matrix/".
    """
    path = search_folder(root)
    path += root + "/"
    for folder in new_folders:
        path += folder +"/"
        Exist = os.path.exists(path)
        if not Exist:
            os.makedirs(path)
    return path

def show_line(line, name):
    """!
    show_line function, show the graph of a line with sorted value.

    @param line the list of value to put in the graph.
    @param name the name of the line in the graph.

    @return nothing.
    """
    fig, ax = plt.subplots()
    ax.plot([i for i in range(len(line))], sorted(line), 'r-', label=name)

    ax.grid(linestyle='--')
    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
    plt.show()

def calcul_EER_lines(FRR, FAR, max_threshold):
    """!
    calcul_EER_lines function, with value of distance obtain in FAR and FRR calculate the far and frr percent authentication.

    @param FRR all the distance between all the couple with the same user.
    @param FAR all the distance between all the couple with a different user.
    @param max_threshold number of treshold to calculate the percent of authentification for each threshold. Greater he is more accurate is the result but the time of execution is greater too.

    @return the two list of authentification percent for all threshold, for false acceptance and false reject.
    """
    frr = []
    far = []
    for threshold in range(max_threshold):
        num_frr = 0
        for fr in FRR:
            if fr > threshold/max_threshold:
                num_frr += 1
        num_far = 0
        for fa in FAR:
            if fa < threshold/max_threshold:
                num_far += 1
        frr.append((num_frr/len(FRR)*100))
        far.append((num_far/len(FAR)*100))
    return frr,far

def show_EER(frr, far, max_threshold, title, eer, threshold):
    """!
    show_EER function, make a graph of the FAR, FRR, EER and threshold optimum for the authentification.
    Show the graph and save it too in the folder of the main file who execute this function.

    @param frr the percent of reject for all threshold with couple with the same user
    @param far the percent of authentification for all threshold with couple with diffferent user
    @param max_threshold the number of threshold use for calculate far and frr. Not necessary gonna be delete in the future.
    @param title the title of the graph and the file.
    @param eer the value of the EER.
    @param threshold the value use by the system to accept authentification.

    @return nothing.
    """
    fig, ax = plt.subplots()
    max_threshold = len(far)
    ax.plot([i/max_threshold*100 for i in range(max_threshold)], frr, 'g-', label='FRR')
    ax.plot([i/max_threshold*100 for i in range(max_threshold)], far, 'r-', label='FAR')
    ax.hlines(eer,0,100, 'b',  label="EER "+str(eer)+"%")
    ax.vlines(threshold,0,100, 'k',  label="Treshold "+str(threshold)+"%")
    # ax.plot([i/max_threshold*100 for i in range(max_threshold)], [abs(far[i]-frr[i]) for i in range(len(frr))], 'b-', label='Distance')

    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
    ax.grid(linestyle='--')
    fig.set_size_inches(18.5, 10.5, forward=True)
    plt.title(title)
    plt.xlabel('Distance (%)')
    plt.ylabel('Rate (%)')

    path = search_folder("result")
    file = Path(path+"result/EER_template/"+title+".png")
    plt.savefig(title+".png")
    plt.show()
    plt.close()

def print_EER(index, max_threshold, frr, far):
    """!
    print_EER function, Print the value of false reject, false acceptance, distance and the mean between the far and frr for the threshold chosen and one before and after.

    @param index the index of the optimum threshold calculate.
    @param max_threshold number of threshold calculate. Not necessary gonna be delete in the future.
    @param frr the percent of reject for all threshold with couple with the same user
    @param far the percent of authentification for all threshold with couple with diffferent user

    @return nothing.
    """
    max_threshold = len(far)
    print(" threshold-1 : ", (index-1)/max_threshold*100, end =" ")
    print(" value frr-1 :", round(frr[index-1],2), end =" ")
    print("value far-1 :", round(far[index-1],2), end =" ")
    print("distance :", round(abs(far[index-1] - frr[index-1]),2), end =" ")
    print("EER :", round(abs(far[index-1] + frr[index-1])/2,2))

    print(" threshold   : ", index/max_threshold*100, end =" ")
    print(" value frr   :", round(frr[index],2), end =" ")
    print("value far   :", round(far[index],2), end =" ")
    print("distance :", round(abs(far[index] - frr[index]),2), end =" ")
    print("EER :", round(abs(far[index] + frr[index])/2,2))

    print(" threshold+1 : ", (index+1)/max_threshold*100, end =" ")
    print(" value frr+1 :", round(frr[index+1],2), end =" ")
    print("value far+1 :", round(far[index+1],2), end =" ")
    print("distance :", round(abs(far[index+1] - frr[index+1]),2), end =" ")
    print("EER :", round(abs(far[index+1] + frr[index+1])/2,2))

def exchange(list, i, j):
    """!
    exchange function, exchange two value in a list.

    @param list the list.
    @param i the first index.
    @param j the second index.

    @return nothing.
    """
    temp = list[i]
    list[i] = list[j]
    list[j] = temp

def show_image(image):
    """!
    show_image function, show the image given. This function has to be change the color of the image aren't good, this function still functional.

    @param image the image to be show.

    @return nothing.
    """
    # img = mpimg.imread('your_image.png')
    # im_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.show()
    imgplot = plt.imshow(image)
    plt.show()

def retrieve_first_row_from_file(file):
    """!
    retrieve_first_row_from_file function, return the first row of the file given. Not use for the moment.

    @param file the path to the file.

    @return the first row of the file.
    """
    f = open(file, "r")
    reader = csv.reader(f, quoting = csv.QUOTE_NONNUMERIC, delimiter = ' ')
    for row in reader:
        first_line = row
        f.close()
        return first_line

def retrieve_tab_from_file(file):
    """!
    retrieve_tab_from_file function, return all the row of the file given in a list.

    @param file the path to the file.

    @return the list with all row of the file.
    """
    f = open(file, "r")
    reader = csv.reader(f, quoting = csv.QUOTE_NONNUMERIC, delimiter = ' ')
    res = []
    for row in reader:
        res.append(row)
    f.close()
    return res

def save_plot(tab, file):
    """!
    save_plot function, save a graph in a name file given with the data store in a list. Use in the write function.
    Actually work only with a graph with only one line. Not very usefull at the moment.

    @param tab the data use to make the graph. exemple of list[line, abscisse=[], name_line, title, xlabel, ylabel].
    @param file the path to save the graph.
    """
    line = tab[0]
    abscisse = [i for i in range(len(line))] if tab[1] == [] else tab[1]
    name = tab[2]
    title = tab[3]
    x = tab[4]
    y = tab[5]
    fig, ax = plt.subplots()
    ax.plot(abscisse, sorted(line), 'r-', label=name)
    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

    ax.grid(linestyle='--')
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    fig.set_size_inches(18.5, 10.5, forward=True)
    plt.savefig(file)
    plt.close()

def find_nb_file(path, name_file):
    """!
    find_nb_file function, find the number of file with the same name without the number in suffix. Use in write function.
    For name file image.bmp 0_image.bmp gonna increse the count.

    @param path the path for the folder in a string.
    @param name_file the name file to count the number of file. Exemple image.bmp.

    @return the number of file
    """
    files = all_images_name(path)
    count = 0
    for file in files:
        # print(file , name_file)
        number = split('_| |\.', file)
        # print(file[len(number):] , name_file)
        if file[len(number[0]):] == name_file:
            count += 1
    #         print(file)
    # print(count)
    return count

def find_nb_file_avaible(path, name_file):
    """!
    find_nb_file_avaible function, find a number suffix not use in the folder for the name file given. Use in write function.
    For name file image.bmp and 0_image.bmp, 1_image.bmp, 3_image.bmp return 2 is avaible

    @param path the path for the folder in a string.
    @param name_file the name file to count the number of file. Exemple image.bmp.

    @return a number suffix avaible.
    """
    files = all_images_name(path)
    count = 0
    file = str(count)+name_file
    path_file = Path(path+file)
    while os.path.exists(path_file):
        count += 1
        file = str(count)+name_file
        path_file = Path(path+file)
    return count

def save_image_attacker_and_target(path, attacker, target):
    """!
    save_image_attacker_and_target function, save the original digital print of the attacker and the target in the folder. Use in write function.

    @param path the path for the folder in a string.
    @param attacker Image object of the attacker.
    @param target Image object of the target.
    """
    if attacker != None:
        image_attacker = Path(path+"attacker.bmp")
        if not os.path.exists(image_attacker):
            imageio.imwrite(image_attacker, attacker.image)
    if target != None:
        image_target= Path(path+"target.bmp")
        if not os.path.exists(image_target):
            imageio.imwrite(image_target, target.image)


def write(tab, new_folder, attacker=None, target=None, name="", extention=".csv", number_file=-1):
    """!
    write function, use for save data, image or graph depending of the extention in the parameter.

    @param tab the data, image or graph to save.
    @param new_folder list of folder to save the contained.
    @param attacker Image object of the attacker. Default as None.
    @param target Image object of the target. Default as None.
    @param name of the file after the number suffix. Default value "".
    @param extention the extention of the file create define also the type of file. Default value ".csv"
    @param number_file the number use in suffix to the name file. If it is at -1 search a number avaible. Default value -1

    @return the data, image of graph save.
    """
    global switch_folder
    print("write")
    path = create_path("result", new_folder)
    save_image_attacker_and_target(path, attacker, target)
    path = create_path(path, switch_folder[extention])
    if number_file == -1:
        # nb_file = str(find_nb_file(path, name+extention))
        nb_file = str(find_nb_file_avaible(path, name+extention))
    else:
        nb_file = str(number_file)
    file = Path(path+nb_file+name+extention)
    if extention == ".csv":
        f = open(file, "w")
        writer = csv.writer(f, quoting = csv.QUOTE_NONNUMERIC, delimiter = ' ')
        if isinstance(tab[0], int) or isinstance(tab[0], float):
            writer.writerow(tab)
        else:
            for row in tab:
                writer.writerow(row)
        print("\n data")
        f.close()
    if extention == ".bmp":
        imageio.imwrite(file, tab)
        print("\n image")
    if extention == ".png":
        save_plot(tab, file)
        print("\n plot")

    print(file,"\n")
    return tab

def graph_multiple_parameter(folder, start_distance, tab_parameter, min_mean_max, time, xlabel, name, number_file=-1, quantile=0):
    """!
    graph_multiple_parameter function, use for make and save graph with data of result obtain with different parameter.

    @param folder list of string with the folders to save graph.
    @param start_distance the distance template before the attack and the modification of the image attaker.
    @param tab_parameter list of different parameter use. Exemple with delta parameter [10,50,100,200].
    @param min_mean_max list of three list the first is the min value obtain in the result, the seond is the mean value and the third is the max value.
    @param time list of time execution.
    @param xlabel string to put in the xlabel of the graph. Example "Parameter Delta"
    @param name the name of the graph.
    @param quantile percent of quantile to delete extreme value. Exemple 0.01 will delete the min 1% and the max 1% value to keep 98% of the values. Not use for the moment.
    """
    path = create_path("result", folder)
    path = create_path(path, ["graph_multiple_parameter"])
    # nb_file = str(find_nb_file(path, name+".png"))
    if number_file == -1:
        nb_file = str(find_nb_file_avaible(path, name+".png"))
    else:
        nb_file = str(number_file)

    fig, ax = plt.subplots()
    min_plot = ax.plot(tab_parameter, min_mean_max[0], 'r-', marker = 'o', label = "distance min")
    mean_plot = ax.plot(tab_parameter, min_mean_max[1], 'b-', marker = 'o', label = "distance moyenne")
    max_plot = ax.plot(tab_parameter, min_mean_max[2], 'g-', marker = 'o', label = "distance max")
    ax2 = ax.twinx()
    time_plot = ax2.plot(tab_parameter, time, 'y-', marker = 'o', label = "time")
    ins = min_plot+mean_plot+max_plot+time_plot
    labs = [l.get_label() for l in ins]
    ax.legend(ins, labs, loc="upper center")
    ax.grid(linestyle='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(" Distance ")
    ax.set_ylim([0, start_distance])
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel(" Temps ")
    plt.title(name)
    fig.set_size_inches(18.5, 10.5, forward=True)

    plt.savefig(path+nb_file+name+".png")
    print("\n plot parameter")
    print(path+nb_file+name+".png","\n")

    # plt.show()
    plt.close()

def set_heatmap_grid(nb_images, plot):
    """!
    set_heatmap_grid function, set the grid in the heatmap to separate same user and different user in the heatmap.
    Use for database with 8 image per user. Have to be modified for change that.

    @param nb_images the number of image in the database.
    @param plot the plot of the heatmap.
    """
    color_user = "black"
    color_same_user = "black"
    for i in range(1, int(nb_images/8)):
        plot.hlines(i*8, 0, nb_images, colors=color_user, linestyles="dashed")
        plot.vlines(i*8, 0, nb_images, colors=color_user, linestyles="dashed")

    for i in range(0, int(nb_images/8)):
        plot.hlines(i*8, i*8, (i+1)*8, colors=color_same_user)
        plot.vlines(i*8, i*8, (i+1)*8, colors=color_same_user)

        plot.hlines((i+1)*8, i*8, (i+1)*8, colors=color_same_user)
        plot.vlines((i+1)*8, i*8, (i+1)*8, colors=color_same_user)

    plot.hlines(0, 0, nb_images, colors="black")
    plot.vlines(0, 0, nb_images, colors="black")
    plot.hlines(nb_images, 0, nb_images, colors="black")
    plot.vlines(nb_images, 0, nb_images, colors="black")

def heatmap_time_execution(time_couple, images, time_limit, subtitle=""):
    """!
    heatmap_time_execution function, make the heatmap of the time of execution. Make for the quadratic model this functionmust be modified for other attack.
    Set "skyblue" for time < 1sec, "green" the attack is out of time(time_limit+1), "black" if the solution is infeasable (time_limit+2) and a degraded of red for the rest.

    @param time_couple list in two dimension of all time of execution of every couple in the database.
    @param images list of Image object.
    @param time_limit the time limit accepted by the attack.
    @param subtitle string add to the start of the title. Default value "".
    """
    df = pd.DataFrame(np.array(time_couple))
    # time_couple_np = np.array(time_couple)
    # max_value = time_couple_np
    fig, ax = plt.subplots()
    size_plot = 10.5
    fig.set_size_inches(size_plot, size_plot, forward=True)

    cmap_reds = plt.get_cmap('Reds')
    num_colors = time_limit + 2

    instant_calcul_color = ['skyblue']
    degraded_color = [cmap_reds(i / num_colors) for i in range(1, num_colors-2)]
    out_of_time_color = 'green'
    infeasable_solution_color = 'black'
    colors = instant_calcul_color + degraded_color + [out_of_time_color, infeasable_solution_color]
    # colors = [infeasable_solution_color, out_of_time_color] + instant_calcul_color + degraded_color

    cmap = LinearSegmentedColormap.from_list('', colors, num_colors)
    ax = sns.heatmap(df, cmap=cmap, vmin=0, vmax=num_colors, square=True, cbar=True)
    # plt.colorbar(ax.collections[0], ticks=range(num_colors + 1))
    set_heatmap_grid(len(time_couple), ax)
    size_str = "Heatmap time, images size : " + str(images[0].n)+"x"+str(images[0].m)
    bdd_str = ", BDD : " + str(images[0].bdd)
    template_str = ", Size Template : "+str(images[0].size_template)
    time_str = ", Time limit : "+str(time_limit)
    title = subtitle + size_str + bdd_str + template_str + time_str
    plt.title(title)
    plt.xlabel(" Target ")
    plt.ylabel(" Attacker ")

    if not os.path.exists("heatmap/"):
        os.makedirs("heatmap/")
    plt.savefig("heatmap/"+title+".png")
    plt.show()
    plt.close()

def heatmap_distance(distance_couple, images, threshold=-1, subtitle=""):
    """!
    heatmap_distance function, make show and save the heatmap of the distance given.

    @param distance_couple list in two dimension of the distance between couple. Can use distance image, template, before or after an attack.
    @param images list of Image object.
    @param threshold the value use by the system to accept authentification. Default value -1 set "skyblue" color for the distance equal to 0. if != -1 set the distance < threshold in "skyblue" color.
    @param subtitle string add to the start of the title. Default value "".
    """
    df = pd.DataFrame(np.array(distance_couple))
    distance_couple_np = np.array(distance_couple)
    max_value = distance_couple_np.max()
    fig, ax = plt.subplots()
    size_plot = 10.5
    fig.set_size_inches(size_plot, size_plot, forward=True)

    size_str = "Heatmap distance, images size : " + str(images[0].n)+"x"+str(images[0].m)
    bdd_str = ", BDD : " + str(images[0].bdd)
    template_str = ", Size Template : "+str(images[0].size_template)


    cmap_reds = plt.get_cmap('Reds')
    num_colors = 1000
    title = subtitle + size_str + bdd_str + template_str
    if threshold == -1:
        threshold_color = 1
    else:
        threshold_color = int(threshold / max_value*1000)
        threshold_str = ", Threshold " + str(threshold)
        title += threshold_str
    autentification_succeeded_color = ['skyblue' for i in range(threshold_color)]
    degraded_color = [cmap_reds(i / num_colors) for i in range(threshold_color, num_colors-1)]
    out_of_time_color = 'green'
    infeasable_solution_color = 'black'
    colors = autentification_succeeded_color + degraded_color

    cmap = LinearSegmentedColormap.from_list('', colors, num_colors)
    ax = sns.heatmap(df, cmap=cmap, vmin=0, vmax=max_value, square=True, cbar=True)
    # plt.colorbar(ax.collections[0], ticks=range(num_colors + 1))
    set_heatmap_grid(len(distance_couple), ax)
    plt.title(title)
    plt.xlabel(" Target ")
    plt.ylabel(" Attacker ")
    if not os.path.exists("heatmap/"):
        os.makedirs("heatmap/")
    plt.savefig("heatmap/"+title+".png")
    plt.show()
    plt.close()

def show_plot(lines_tab, names_tab, title, xlabel, ylabel, abscisse=-1):
    """!
    show_plot function, make show and save the plot create with all the curves given.

    @param lines_tab list of all curves to be add in the plot. A curves is a list of data.
    @param names_tab list of name for all curves.
    @param title the title of the plot.
    @param xlabel the xlabel of the plot.
    @param ylabel the ylabel of the plot.
    @param abscisse the list of value use in abscisse. Default value -1 set the abscisse to the number between 1 and the number of data in the curve.
    """
    colors = ["b","r","g","c","m","y","k"]
    if len(lines_tab) > len(colors) or len(lines_tab) > len(names_tab):
        print(" Wrong tab for show_plot")
        return
    fig, ax = plt.subplots()
    for i in range(len(lines_tab)):
        if abscisse == -1:
            abscisse = [i+1 for i in range(len(lines_tab[i]))]
        ax.plot(abscisse, lines_tab[i], colors[i]+"-", marker = 'o', label = names_tab[i])
    plt.title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(linestyle='--')
    # ax.set_ylim([0,0.2])
    # ax.set_ylim([0,0.2])
    ax.set_ylim(bottom=0)
    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')
    fig.set_size_inches(18.5, 10.5, forward=True)
    # plt.figure(figsize=(18.5, 10.5))
    plt.savefig("save/"+title)
    plt.show()
    plt.close()

def add_plot(lines_tab, names_tab, ax, i, j, ylabel, abscisse=-1):
    """!
    add_plot function, add plot in the subplot given. Use in make_subplot function.

    @param lines_tab list of all curves to be add in the sub plot. A curves is a list of data.
    @param names_tab list of name for all curves.
    @param ax the main plot.
    @param i the line in the main plot for the new sub plot.
    @param j the column in the main plot for the new sub plot.
    @param ylabel the ylabel of the plot.
    @param abscisse the list of value use in abscisse. default value -1 set the abscisse to the number between 1 and the number of data in the curve.
    """
    colors = ["b","r","g","c","m","y","k"]
    if len(lines_tab) > len(colors) or len(lines_tab) > len(names_tab):
        print(" Wrong tab for show_plot")
        return
    if i*2 +j < len(names_tab):
        name_label = names_tab[i*2 +j]
    else:
        name_label=""
    for k in range(len(lines_tab)):
        if abscisse == -1:
            abscisse = [l for l in range(len(lines_tab[k]))]
        if i*2 +j == k:
            ax[i][j].plot(abscisse, lines_tab[k], colors[k]+"-", marker = 'o', label = name_label)
            ax[i][j].legend(loc='lower right', shadow=True, fontsize='x-large')
        else:
            ax[i][j].plot(abscisse, lines_tab[k], colors[k]+"-", marker = 'o')
    if j == 0:
        ax[i][j].set_ylabel(ylabel)
    if i == len(ax)-1:
        ax[i][j].set_xlabel("Parameter")

def make_subplot(lines_tab, names_tab, main_title, title_plot, xlabel, ylabel, abscisse=-1, ylim=-1):
    """!
    make_subplot function, make, show and save a subplot 2 by 2 with curves store in a list.
    This fucntion was create to show all the graph of graph_multiple_parameter easier.
    If it's necessary a complete exemple of how use this function is in the function test_parameter in the genetique file.

    @param lines_tab list of all plot curve to be add in the main plot. A plot curve is a list of curve. A curves is a list of data.
    @param names_tab list of name for all curves.
    @param main_title the title of the main plot.
    @param title_plot the list of title for all the sub plot.
    @param xlabel the xlabel of the plot. Not use for the moment.
    @param ylabel the ylabel of the plot.
    @param abscisse the list of abscisse use in the different sub_plot. Abscisse contain a list of value use in abscisse or -1. Default value -1 set the abscisse to the number between 1 and the number of data in the curve.
    """
    fig, ax = plt.subplots(2, 2)
    for i in range(2):
        for j in range(2):
            if abscisse == -1:
                add_plot(lines_tab[i*2+j], names_tab, ax, i, j, ylabel)
            else:
                add_plot(lines_tab[i*2+j], names_tab, ax, i, j, ylabel, abscisse[i*2+j])
    fig.set_size_inches(18.5, 10.5, forward=True)
    fig.suptitle(main_title)

    for i, sub_ax in enumerate(ax.flat):
        # sub_ax.set(ylabel=ylabel)
        sub_ax.set_title(title_plot[i])
        sub_ax.grid(linestyle='--')

        sub_ax.set_ylim(bottom=0)
        if ylim !=-1:
            sub_ax.set_ylim(top=ylim)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for sub_ax in ax.flat:
    #     sub_ax.label_outer()
    folder = ["genetic", "subplot"]
    path = create_path("result", folder)
    plt.savefig(path+main_title)
    print("\n subplot")
    print(path+main_title)
    # plt.show()
    plt.close()

def graph_multiple_parameter_file(folder, start_distance, tab_parameter, number, parameter_list, xlabel, name, line_solution=0,quantile=0):
    """!
    graph_multiple_parameter_file function, retrieve name file and data for use graph_multiple_parameter function.
    If it's necessary a complete exemple of how use this function is in the function test_parameter in the genetique file.

    @param folder list of string with the folders to save graph.
    @param start_distance the distance template before the attack and the modification of the image attaker.
    @param tab_parameter list of different parameter use. Exemple with delta parameter [10,50,100,200].
    @param number the number suffix use to retrieve file.
    @param parameter_list a list of parameter use to save the file. for genetic a parameter is [max_while, number_select, proba_mutation, weightI] and parameter_list is a list of this.
    @param xlabel string to put in the xlabel of the graph. Example "Parameter Delta"
    @param name the name of the graph.
    @param line_solution the line retrive in the file. Default value 0 is the line of the distance image, 1 is the distance template.
    @param quantile percent of quantile to delete extreme value. Exemple 0.01 will delete the min 1% and the max 1% value to keep 98% of the values.
    """
    new_folder = folder[:]
    new_folder.append("data")
    path = create_path("result", new_folder)
    print(path)
    file_list = create_file_data_name_list("genetic", number, parameter_list)
    # file_list = create_file_data_name_list("gradiant", number, parameter_list)

    line_min = []
    line_mean = []
    line_max = []
    line_time = []
    for file in (file_list):
        tab_from_file = (retrieve_tab_from_file(path+file))
        tab_solution = pd.Series(tab_from_file[line_solution])
        tab_time = pd.Series(tab_from_file[-1])
        tab_solution = tab_solution[tab_solution.between(tab_solution.quantile(quantile), tab_solution.quantile(1-quantile))]
        tab_time = tab_time[tab_time.between(tab_time.quantile(quantile), tab_time.quantile(1-quantile))]
        line_min.append(min(tab_solution))
        line_mean.append(mean(tab_solution))
        line_max.append(max(tab_solution))
        line_time.append(mean(tab_time))
    if quantile != 0:
        name += (", top " + str(quantile*100) + "% extreme supprimer")
    graph_multiple_parameter(folder, start_distance, tab_parameter, [line_min,line_mean,line_max], line_time, xlabel, name, number)
    return [line_min,line_mean,line_max], line_time

def create_file_data_name_list(name_algorithm, number, parameter_list):
    """!
    create_file_data_name_list function, create a list of name file for load data of multiple attack by genetic algorithm.
    Exemple
    create_file_data_name_list("genetic", 0, [[5,5,0.03,0.1],[5,10,0.03,0.1],[5,15,0.03,0.1],[5,20,0.03,0.1]])
    create_file_data_name_list("gradiant", 0, [[0,100],[0.1,100],[[0.2,100],[[0.3,100],[0.4,100])

    @param name_algorithm the name of the algorithm use.
    @param number the number suffix use to retrieve file.
    @param parameter_list a list of parameter use to save the file. A parameter is [max_while, number_select, proba_mutation, weightI] and parameter_list is a list of this.
    The function create_parameter_list was create to create this.

    @return the list of name file create.
    """
    solution_file_list = []
    for p in parameter_list:
        solution_file_list.append(create_file_data_name(name_algorithm, number, p, add_number_file=True, add_extension=True))
    return solution_file_list

def create_file_data_name(name_algorithm, number_file, parameter, add_number_file=False, add_extension=False):
    """!
    create_file_data_name function, create a name file for save data of multiple attack by algorithm gives in name_algorithm parameter.
    Exemple
    create_file_data_name("genetic", 0, [5,10,0.03,0.1], True, True)
    create_file_data_name("gradiant", 0, [0.1,100], True, True)

    @param name_algorithm the name of the algorithm use.
    @param number the number suffix use to retrieve file.
    @param parameter a list of parameter use to save the file. The list depend of the algorithm use, genetic have 4 parameter, gradiant 2.
    @param add_number_file boolean to know if the number_suffix is add at the start of the file. Default value False.
    @param add_extension boolean to know if the extension is add at the end of the file. Default value False.

    @return the list of name file create.
    """
    name = ""
    if add_number_file:
        name += str(number_file)
    if name_algorithm == "genetic":
        name += ("_max_"+str(parameter[0]))
        name += ("_population_"+str(parameter[1]*parameter[1]))
        name += ("_proba_"+str(int(parameter[2]*100)))
        name += ("_weightI_"+str(int(parameter[3]*100)))
    elif name_algorithm == "gradiant":
        name += "_weightI_"+str((parameter[0]))
        name += "_delta_"+str(parameter[1])
    else:
        print(" Error in create_file_data_name function bad name_algorithm :",name_algorithm)
    if add_extension:
        name += ".csv"
    return name

def create_file_image_name(weightI, start_distance_template, person):
    """!
    create_file_image_name function, create a name file for save image.

    @param weightI the weight of the image in the objective function.
    @param start_distance_template the distance template before running any attack.
    @param person a Person object of the result of the attack.

    @return the name file create.
    """
    name_file = "_weightI_" + str(weightI)
    name_file += "_start_" + str(round(start_distance_template,4))
    name_file += "_template_" + str(round(person.distance_template,4))
    name_file += "_image_" + str(round(person.distance_image,4))
    name_file += "_objective_" + str(round(person.objective,4))
    return name_file

def create_parameter_list(tab, index, constant_parameter_list):
    """!
    create_parameter_list function, create a list of parameter_list to be used in graph_multiple_parameter_file function.
    Exemple : create_parameter_list([5,10,15,20], 1, [5,None,0.03,0.1]) return [[5,5,0.03,0.1],[5,10,0.03,0.1],[5,15,0.03,0.1],[5,20,0.03,0.1]]

    @param tab the list of the variation of one parameter. Exemple use for number_select [5,10,15,20].
    @param index the index of the parameter use in tab. Exemple number_select index 1 for genetic algorithm. See create_file_data_name function to know the different index.
    @param constant_parameter_list the list of value for other parameter the value in the index has no importance. Exemple [5,None,0.03,0.1].

    @return the list of parameter create.
    """
    parameter_list = []
    for i in range(len(tab)):
        temp = []
        for j in range(len(constant_parameter_list)):
            if index == j:
                temp.append(tab[i])
            else:
                temp.append(constant_parameter_list[j])
        parameter_list.append(temp[:])
    return parameter_list

def distance_template_norm(T1, T2):
    """!
    distance_template_norm function, calculate the distance normalize between two template. The distance is the sum of the difference.

    @param T1 the first template.
    @param T2 the second template.

    @return the distance calculate.
    """
    distance = 0
    for t1, t2 in zip(T1,T2):
        distance += abs(t1 - t2)
    return distance/len(T1)

def distance_image_norm(I1, I2):
    """!
    distance_image_norm function, calculate the distance normalize between two image. The distance is the sum of the difference.

    @param I1 the first image.
    @param I2 the second image.

    @return the distance calculate.
    """
    distance = 0
    max = 0
    for row1, row2 in zip(I1, I2):
        for i1, i2 in zip(row1, row2):
            distance += abs(int(i1) - int(i2))
            max += 255
    return distance/max

def distance_feature_norm(F1, F2, width):
    """!
    distance_feature_norm function, calculate the distance normalize between two feature. The distance is the square root of the sum of the square of the difference.

    @param F1 the first feature.
    @param F2 the second feature.

    @return the distance calculate.
    """
    distance = 0
    max = pow(255*len(F1),2) * width
    for f1, f2 in zip(F1,F2):
        distance += pow((f1 - f2), 2)
    distance = sqrt(distance)/sqrt(max)

    return (distance)

def sobel_filter_no_class(im):
    """!
    sobel_filter_no_class function, make the sobel filter on the image to make a vector of feature.

    @param im the image.

    @return the vector of feature calculate.
    """
    n, m = (np.array(im).shape)
    fa = np.zeros((n, m))
    new_image_x = np.zeros((n, m))
    new_image_y = np.zeros((n, m))
    scipy.ndimage.sobel(im, axis=0, output=new_image_x, cval=0.0)
    scipy.ndimage.sobel(im, axis=1, output=new_image_y, cval=0.0)
    fa = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    if fa.max() == 0:
        print(" All the pixels of the image are white")
    F = [sum(fa[i][j] for j in range(m)) for i in range(n)]
    return F

def projection_no_class(F, M):
    """!
    projection_no_class function, make the projection with the vector of feature and the matrix of the user.

    @param F the vector of feature.
    @param F2 the matrix of the user.

    @return the result obtain.
    """
    r = np.dot(F, M)
    return r

def binarisation_no_class(R):
    """!
    binarisation_no_class function, calculate the template with the result obtain after projection.

    @param R the result obtain after projection.

    @return the template calculate.
    """
    T = [0] * len(R)
    for i in range(len(R)):
        T[i] = 0 if R[i] < 0 else 1
    return T

def template_no_class(im, M):
    """!
    template_no_class function, calculate the template with the image and the matrix of projection.

    @param im the image.
    @param M the matrix of projection.

    @return the template calculate.
    """
    F = sobel_filter_no_class(im)
    R = projection_no_class(F, M)
    T = binarisation_no_class(R)
    return T

def copy_home(tab):
    """!
    copy_home function, copy_home make a copy of a two dimension list.

    @param tab the two dimension list.

    @return the copy make.
    """
    new = [[0 for j in range(len(tab[0]))] for i in range(len(tab))]
    for i in range(len(tab)):
        for j in range(len(tab[0])):
            new[i][j] = tab[i][j]
    return new
