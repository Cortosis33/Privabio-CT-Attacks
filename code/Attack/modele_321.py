#!/usr/bin/env python3.7

"""!
File modele_321
Contain the first part of the attack by quadratic model with Gurobi.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from image import *

def make_folder(attacker, target=None):
    """!
    make_folder function, make a list of folders for save data with write function.
    Folder is /modele_321/bdd_***/template_***/image_***x***/.

    @param attacker Image object of the attacker.
    @param target Image object of the target. Default None.  If target is not as None add the folder /attack_***->***/

    @return the list of folder. The list is ["modele_321","bdd_***","template_***", "image_***x***"] for exemple.
    """
    folder = ["modele_321"]
    folder.append("bdd_"+attacker.bdd)
    folder.append("template_"+str(attacker.size_template))

    size_image = str(attacker.n)+"x"+str(attacker.m)
    folder.append("image_"+size_image)
    if target!=None:
        attack = str(attacker.number_image) + "->" + str(target.number_image)
        folder.append("attack_"+attack)
    return folder

def multiple_autenticate_321(images):
    """!
    multiple_autenticate_321 function, make the part 1 of the quadratic model attack for all couple with the same user.

    @param images list of Image object. Normally use all the image of the database.

    @return the list of distance template calculate with the new feature find.
    """
    tab = []
    optimal = [0]*2
    limit = len(images)
    # limit = 8
    for i in range(limit):
        print(" autenticate 321 : ",i)
        min = int(i/8)*8 + i%8 + 1
        max = int(i/8)*8 + 8
        attacker = images[i]
        # attacker_feature = attacker.feature
        for j in range(min, max):
            # print(" autenticate",i,j)
            if (j!=i):
                target = images[j]
                # target_template = target.template
                distance = test_321(attacker, target)
                if distance == -1:
                    optimal[1] += 1
                else:
                    optimal[0] += 1
                    tab.append(distance)
    print(optimal)
    size_image = str(images[0].n)+"x"+str(images[0].m)
    folder = make_folder(images[0])
    # write([tab,optimal], folder)
    # write("modele_321", "autenticate", [tab,optimal], images[0])
    # show_line(tab, "321 autenticate")
    return tab

def multiple_attack_321(images):
    """!
    multiple_attack_321 function, make the part 1 of the quadratic model attack for all couple with a different user.

    @param images list of Image object. Normally use all the image of the database.

    @return the list of distance template calculate with the new feature find.
    """
    tab = []
    optimal = [0]*2
    limit = len(images)
    # limit = 16
    for i in range(limit):
        print(" attack 321 : ",i)
        min = int(i/8+1)*8
        max = len(images)
        # max = 16
        attacker = images[i]
        # attacker_feature = attacker.feature
        for j in range(min, max):
            # print(" attack",i,j)
            if (j!=i):
                target = images[j]
                # target_template = target.template
                distance = test_321(attacker, target)
                if distance == -1:
                    optimal[1] += 1
                else:
                    optimal[0] += 1
                    # print(" ",i,j,":",distance)
                    tab.append(distance)
    print(optimal)
    size_image = str(images[0].n)+"x"+str(images[0].m)
    folder = make_folder(images[0])
    # write([tab,optimal], folder)
    # write("modele_321", "attack", [tab,optimal], images[0])
    # show_line(tab, "321 attack")
    return tab

def test_321(attacker, target):
    """!
    test_321 function, execute the linear part of the quadratic model.

    @param attacker Image object of attacker.
    @param target Image object of target.

    @return the distance between the target template and the template create with the new feature calculate with the model.
    """
    new_fa = modele321(attacker, target)
    # print(res)
    if new_fa == -1:
        return -1
    else:
        # new_ima = copy(attacker)
        # new_ima.change_feature(new_fa)
        # new_ima.change_matrix(target.matrix)
        # distance = distance_template_norm(new_ima.template, target.template)
        proj_321 = projection_no_class(new_fa, target.matrix)
        temp_new = binarisation_no_class(proj_321)
        distance = distance_template_norm(temp_new, target.template)
        return distance

def modele321(attacker, target):
    """!
    modele321 function, construct the linear problem in 3.2.1 of the article
    Authentication Attacks on Projection-based Cancelable Biometric Schemes
    DURBET Axel, GROLLEMUND Paul-Marie, LAFOURCADE Pascal, MIGDAL Denis and THIRY-ATIGHEHCHI
    and return the new feature calculate

    @param attacker Image object of attacker.
    @param target Image object of target.

    @return the new feature find with the model or -1 if the model is infeasable.
    """
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag",0)
        env.start()
        # Create a new model
        model = gp.Model("mip1", env=env)
        # model.Params.NonConvex = 2

        length_image = attacker.n
        width_image = attacker.m
        max_feature_value = ceil(sqrt(1200*1200*2)) * length_image

        M = target.matrix
        T = target.template
        FA = attacker.feature

        # Main variable
        X = model.addMVar(length_image, name="X", lb=0, ub=max_feature_value, vtype=GRB.CONTINUOUS)

        # Variable to make the constraint
        X_FA = model.addMVar(length_image, name="X-FA", lb=-max_feature_value, ub=max_feature_value, vtype=GRB.CONTINUOUS)
        ABS = model.addMVar(length_image, name="ABS", lb=0, ub=max_feature_value, vtype=GRB.CONTINUOUS)

        # Other constant
        K1 = [] # indice where T is equal 0
        K2 = [] # Indice where T is not equal 0

        # Construct K1 and K2
        for i in range(len(T)):
            if T[i] == 0:
                K1.append(i)
            else:
                K2.append(i)
        # Function objective minimize difference betwwen old and new feature
        model.setObjective(sum(ABS[i] for i in range(length_image)), GRB.MINIMIZE)

        # Main Constraint in the model
        for j in K1:
            model.addLConstr(sum(X[i] * M[i][j] for i in range(length_image)) <= pow(10,-12)) # TODO equal ?

        for j in K2:
            model.addLConstr(sum(X[i] * M[i][j] for i in range(length_image)) >= 0)

        # Intermediary constraint
        for i in range(length_image):
            model.addLConstr(X_FA[i] == (X[i]-FA[i]))
            model.addConstr(ABS[i] == gp.abs_(X_FA[i]))

        # Optimize model
        model.optimize()
        # print("\n Model Status :",model.status)
        if model.status != 2:
            return -1
        # for i, v in enumerate(model.getVars()):
        #     print(' %s %g \t' % (v.VarName, v.X),end="")
        vars = model.getVars()
        res = []
        for x in range(length_image):
            # print(" X" + str(x) + " =",vars[x].X,vars[x].VarName)
            res.append(vars[x].X)
        return res


        print('\n Obj: %g\n' % model.ObjVal)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
