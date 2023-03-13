
from function import *
"""!
Image file.
Contains Image class and the function to easily create the instance.
"""
class Image():
    """!
    Class Image.
    Contains attributes and methods needed by different attacks to manipulate images.
    """
    def __init__(self, image, size_template, bdd, name_image):
        """!
        The constructor of the class Image.

        @param image the image of digital print.
        @param size_template the size of the template. Value in [1,∞].
        @param bdd the string of the folder of the database use.
        @param name_image the name of the file of the digital print.
        """
        ## Length of the image
        self.n = len(image)
        ## Width of the image
        self.m = len(image[0])
        ## Two dimension list with value in [0,255] for the value of the pixel. This is the image of the attacker in the attack.
        self.image = image
        ## The size of the template use. Value in [1,∞].
        self.size_template = size_template
        ## The string of the folder of the database use.
        self.bdd = bdd
        ## The name of the file of the digital print.
        self.name_image = name_image
        ## The number of the user of the digital print.
        self.user = retrive_user(name_image)
        ## The vector of feature of the image.
        self.feature = []
        ## The matrix of projection of the user.
        self.matrix = []
        ## The vector of template of the image. It's use to autenticate the user.
        self.template = []
        self.sobel_filter()  # set the feature
        self.retrieve_matrix() # set the matrix
        self.find_template() # set the template

    def change_image(self, image):
        """!
        Method for Image class. change_image function, change the image of digital print and calculate the new feature and template associate.

        @param image the image of the new digital print.
        """
        self.image = image[:]
        self.sobel_filter()
        self.find_template()

    def change_matrix(self, matrix):
        """!
        Method for Image class. change_matrix function, change the matrix and calculate the new template associate.

        @param matrix the new matrix.
        """
        new_matrix = copy_home(matrix)
        self.matrix = new_matrix[:]
        self.find_template()

    def change_feature(self, feature):
        """!
        Method for Image class. change_feature function, change the feature and calculate the new template associate.

        @param feature the new feature.
        """
        self.feature = feature[:]
        self.find_template()

    def retrieve_matrix(self):
        """!
        Method for Image class. retrieve_matrix function, retrive the matrix of a user with a seed obtain with the user, the database, the size of template and the size of digital print use.
        Set the matrix obtain in the object.
        """
        state = random.getstate()
        password = str(self.user)+str(self.bdd)+str(self.size_template)+str(self.n)+str(self.m)
        random.seed(password)
        M = [[random.uniform(-0.5,0.5) for j in range(self.size_template)]for i in range(self.n)]
        random.setstate(state)

        self.matrix = M

    def sobel_filter(self):
        """!
        Method for Image class. sobel_filter function, calculate the sobel_filter of the object.
        Set the feature obtain in the object.
        """
        F = sobel_filter_no_class(self.image)
        self.feature = F
        return F

    def projection(self, F=None):
        """!
        Method for Image class. projection function, calculate the matrix product of the object.

        @param F the feature of the object. Default value None. Depreciate not use anymore.

        @return the result obtain.
        """
        r = projection_no_class(self.feature, self.matrix)
        return r

    def binarisation(self, R):
        """!
        Method for Image class. binarisation function, calculate the binarisation of the object.
        Set the template obtain in the object.

        @param R the result of the matrix product.

        @return the template obtain.
        """
        T = binarisation_no_class(R)
        self.template = T
        return T

    def find_template(self):
        """!
        Method for Image class. find_template function, calculate the template of the object.
        """
        F = self.feature
        R = self.projection(F)
        T = self.binarisation(R)

def create_image(path, image_name, size_template, size_line=-1, size_column=-1):
    """!
    create_image function, retrieve the digital footprint and create the Image object associate.

    @param path the path to find the digital footprint.
    @param image_name the path to find the digital footprint.
    @param size_template the path to find the digital footprint.
    @param size_line the number of line in the wanted image. Default value -1, we take all the line of the entire image.
    @param size_column the number of column in the wanted image. Default value -1, we take all the column of the entire image.

    @return the Image object create.
    """
    image = imageio.v2.imread(path+image_name)
    if size_line == -1:
        line = 0
        line2 = len(image)
    else:
        line = int(len(image)/2-(size_line)/2)
        line2 = line + size_line
    if size_line == -1:
        column = 0
        column2 = len(image[0])
    else:
        column = int(len(image[0])/2-(size_column)/2)
        column2 = column + size_column
    new_image = [[image[i][j] for j in range(column, column2)]for i in range(line, line2)]
    BDD = path.split("/")[-2]
    return(Image(new_image, size_template, BDD, image_name))

def create_all_images(path, size_template, size_line=-1, size_column=-1):
    """!
    create_all_images function, retrieve all digital footprint and create a list of Image object associate.

    @param path the path to find the digital footprint. Exemple : '../../BDD/image/DB1_B/'
    @param size_template the path to find the digital footprint.
    @param size_line the number of line in the wanted image. Default value -1, we take all the line of the entire image.
    @param size_column the number of column in the wanted image. Default value -1, we take all the column of the entire image.

    @return the list of Image object create.
    """
    BDD = path.split("/")[-2]
    images_names = all_images_name(path)
    new_images = []
    for im,image_name in enumerate(images_names):
        new_image = create_image(path, image_name, size_template, size_line, size_column)
        new_images.append(new_image)
    return new_images

def copy(image):
    """!
    copy function, make a copy of a Image object.

    @param image the Image object that we want copy.

    @return the copy of the Image object.
    """
    new_image = image.image.copy()
    new = Image(new_image, image.size_template, image.bdd, image.name_image)
    return new
