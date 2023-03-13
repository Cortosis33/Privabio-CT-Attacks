#!/usr/bin/env python3.7

from modele_321 import *

"""!
File modele_322
Contain the second part of the attack by quadratic model with Gurobi.
"""
def make_folder(attacker, target=None):
    """!
    make_folder function, make a list of folders for save data with write function.
    Folder is /modele_322/bdd_***/template_***/image_***x***/.

    @param attacker Image object of the attacker.
    @param target Image object of the target. Default None.  If target is not as None add the folder /attack_***->***/

    @return the list of folder. The list is ["modele_322","bdd_***","template_***", "image_***x***"] for exemple.
    """
    folder = ["modele_322"]
    folder.append("bdd_"+attacker.bdd)
    folder.append("template_"+str(attacker.size_template))

    size_image = str(attacker.n)+"x"+str(attacker.m)
    folder.append("image_"+size_image)
    if target!=None:
        attack = str(attacker.name_image) + "->" + str(target.name_image)
        folder.append("attack_"+attack)
    return folder

def multiple_autenticate_322(images, objective):
    """!
    multiple_autenticate_322 function, make the quadratic model attack for all couple with the same user.

    @param images list Image object. Normally use all the image of the database.
    @param objective Boolean. Default False, if True the quadratic model add the objective function at the model.

    @return the list of distance template calculate with the new image find.
    """
    tab = []
    optimal = [0]*3
    limit = len(images)
    # limit = 8
    for i in range(limit):
        print(" autenticate 322 : ",i)
        min = int(i/8)*8 + i%8 + 1
        max = int(i/8)*8 + 8
        attacker = images[i]
        for j in range(min, max):
            print(" autenticate 322 : ",i,j)
            if (j!=i):
                target = images[j]
                # if (i == 33 and j == 37):
                    # show_image(attacker_image)
                    # show_image(target_image)
                distance = test_322(attacker, target, objective)
                if distance == -2:
                    optimal[2] += 1
                elif distance == -1:
                    optimal[1] += 1
                else:
                    optimal[0] += 1
                    tab.append(distance)
    print(optimal)
    size_image = str(images[0].n)+"x"+str(images[0].m)
    folder = make_folder(images[0])
    # write([tab,optimal], folder)
    # write("modele_322", "autenticate", [tab,optimal], images[0])
    show_line(tab, "322 autenticate")
    return tab

def all_322(images, objective= False):
    """!
    all_322 function, make the quadratic model attack for all couple.

    @param images list Image object. Normally use all the image of the database.
    @param objective Boolean. Default False, if True the quadratic model add the objective function at the model.

    @return the list of distance template calculate with the new image find and create heatmap of time_execution, distance template and distance image.
    """
    global time_gurobi
    if objective:
        time_gurobi = images[0].n * images[0].m * images[0].m
    else:
        time_gurobi = images[0].n * images[0].m * images[0].m
    tab = []
    optimal = [0]*3
    limit = len(images)
    # size image 10 more than 10 sec
    couple = []
    time_couple = [[ -10 for i in range(80)]for j in range(80)]
    image_distance = [[ -10 for i in range(80)]for j in range(80)]
    template_distance = [[ -10 for i in range(80)]for j in range(80)]
    # limit = 16
    number = 0

    folder_old = "../../result/save_322/progress/data/"
    # retrieve_tab_from_file(folder_old+"0time_couple_2.csv")
    # retrieve_tab_from_file(folder_old+"0template_distance_2.csv")
    # retrieve_tab_from_file(folder_old+"0image_distance_2.csv")
    for i in range(limit):
        print(" attack 322 : ",i)
        min = int(i/8+1)*8
        min = 0
        max = len(images)
        # max = 16
        attacker = images[i]
        for j in range(min, max):
            target = images[j]
            # work = True
            # for c in couple_not_work_10:
            #     if [i,j] == c:
            #         work = False
            if (True):#attacker.user != target.user):#  and i%8 ==j%8):
                print(" attack 322 : ",i,j)
                # print(number)
                # if (number == 50):
                #     return
                number+=1
                status, time_optimisation, objective_value, new_ima, distance = test_322(attacker, target, objective)
                if status == -2: # 3.2.2 infeasable
                    print("3.2.2 infeasable")
                    optimal[2] += 1
                    time_couple[i][j] = time_gurobi + 1
                    image_distance[i][j] = objective_value
                    template_distance[i][j] = distance
                elif status == -1: # 3.2.1 infeasable
                    print("3.2.1 infeasable")
                    optimal[1] += 1
                elif status == -3: # out of time
                    print("3.2.2 out of time")
                    time_couple[i][j] = time_gurobi
                    image_distance[i][j] = objective_value
                    template_distance[i][j] = distance
                elif status == 1: # solution find
                    # print(" attack 322 : ",i,j)
                    print("3.2.2 optimal find")
                    time_couple[i][j] = distance[1]
                    image_distance[i][j] = objective_value
                    template_distance[i][j] = distance
                    optimal[0] += 1
                    tab.append(distance)
                else:
                    print("\n    ERROR\n")
            else:
                time_couple[i][j] = -1
            print("time_couple", time_couple[i][j])
            print("image_distance", image_distance[i][j])
            print("template_distance", template_distance[i][j])

        write(time_couple, ["save_322","progress"], name="time_couple_"+str(i))
        write(image_distance, ["save_322","progress"], name="image_distance_"+str(i))
        write(template_distance, ["save_322","progress"], name="template_distance_"+str(i))
    heatmap_distance(image_distance, images, subtitle="image_distance_")
    heatmap_distance(template_distance, images, subtitle="template_distance_")
    # print()
    # print(couple)
    # print()
    # print_tab(couple)
    # print(optimal)

    heatmap_time_execution(time_couple, images, time_gurobi)

    folder = make_folder(images[0])

    size_str = " Images size : " + str(images[0].n)+"x"+str(images[0].m)
    bdd_str = " BDD : " + str(images[0].bdd)
    template_str = " Size Template : "+str(images[0].size_template)
    time_str = " Time limit : "+str(time_gurobi)
    title = size_str + bdd_str + template_str + time_str
    write(time_couple, folder, name=" Time "+title)
    # write([tab,optimal], folder)
    # write("modele_322", "attack", [tab,optimal], images[0])
    # show_line(tab, "322 attack")
    return tab

def test_322(attacker, target, objective= False):
    """!
    test_322 function, execute the quadratic model.

    @param attacker Image object of attacker.
    @param target Image object of target.
    @param objective Boolean. Default False, if True the quadratic model add the objective function at the model.

    @return the distance between the target template and the template create with the new feature calculate with the model.
    """
    new_feature = modele321(attacker, target)
    # print("f",new_feature)
    # print("f",target.feature)
    if new_feature == -1:
        return -1
    else:
        # print(new_feature)
        # print(attacker.name_image, target.name_image)
        res = modele322(attacker, new_feature, objective)
        print_tab(res)
        status, time_optimisation, objective_value, new_ima = res
        if new_ima == None:
            distance = None
        else:
            new_ima = np.array(new_ima, dtype=np.uint8)
            # print("f",sobel_filter_no_class(new_ima))
            # show_image(new_ima)
            # show_image(attacker.image)
            # show_image(target.image)
            # proj_321 = projection_no_class(new_feature, target.matrix)
            # temp_new = binarisation_no_class(proj_321)
            # print("d", distance_template_norm(target.template, temp_new))
            attacker_template = template_no_class(new_ima, target.matrix)
            distance = distance_template_norm(target.template, attacker_template)
            folder = make_folder(attacker)
            title = "_" + str(attacker.name_image) + "->" + str(target.name_image)
            title += "_template_"
            title += str(distance)
            title += "_objective_"
            title += str(objective_value)
            write(new_ima, folder, name=title, extention=".bmp")
        return [status, time_optimisation, objective_value, new_ima, distance]
        # new_image = Image(new_ima, attacker.size_template, attacker.bdd, attacker.name_image)
        # show_image(image)
        # print_tab(new_image)
        # show_image(new_image)
        attacker_template = template_no_class(new_ima, target.matrix)
        distance = distance_template_norm(target.template, attacker_template)
        # print(" distance", distance)
        return distance

def modele322(image, FA, objective=False):
    """!
    modele322 function, construct the quadratic problem in 3.2.2 of the article
    Authentication Attacks on Projection-based Cancelable Biometric Schemes
    DURBET Axel, GROLLEMUND Paul-Marie, LAFOURCADE Pascal, MIGDAL Denis and THIRY-ATIGHEHCHI
    and return the new image calculate

    @param attacker Image object of attacker.
    @param target Image object of target.
    @param objective Boolean. Default False, if True the quadratic model add the objective function at the model.

    @return a list with [code, time_execution, objective_value, the image find]. code is 1 if optimal find, -2 if infeasable or -3 if the the model is out of time.
    """
    try:
        global time_gurobi
        print(time_gurobi)
        env = gp.Env(empty=True)
        env.setParam("OutputFlag",0)
        env.start()
        start = time()
        # Create a new model
        model = gp.Model("mip1", env=env)
        model.Params.NonConvex = 2
        model.params.TimeLimit = time_gurobi

        Ia = image.image
        #
        length_image = image.n
        width_image = image.m

        max_filtered_value = 1141 # pow(10,6)#sqrt((1200*1200)*2) 1141

        # Main variable
        X = model.addMVar((length_image+2, width_image+2), name="X", lb=0, ub=255, vtype=GRB.INTEGER)

        # Variable for intermediate constraint
        ALPHA = model.addMVar((length_image, width_image), name="ALPHA",lb=-1200, ub=1200, vtype=GRB.CONTINUOUS)
        BETA = model.addMVar((length_image, width_image), name="BETA",lb=-1200, ub=1200, vtype=GRB.CONTINUOUS)
        filtered = model.addMVar((length_image, width_image), name="Fa", lb=0, ub=max_filtered_value, vtype=GRB.CONTINUOUS) # ub =255 ?

        ABS = model.addMVar((length_image, width_image), name="Abs", lb=0, ub=255, vtype=GRB.CONTINUOUS)
        X_IA = model.addMVar((length_image, width_image), name="X_IA",lb = -255, ub =255, vtype=GRB.CONTINUOUS)

        # Objective function
        if objective:
            model.setObjective(sum(ABS[i][j] for i in range(length_image) for j in range(width_image)), GRB.MINIMIZE)

        # Main constraint
        for i in range(1,length_image+1):
            model.addLConstr(sum(filtered[i-1][j-1] for j in range(1,width_image+1)) == FA[i-1])
            for j in range(1,width_image+1):
                model.addLConstr(ALPHA[i-1][j-1] == X[i-1][j-1] + 2*X[i][j-1] + X[i+1][j-1]
                                                -X[i-1][j+1] - 2*X[i][j+1] - X[i+1][j+1])
                model.addLConstr(BETA[i-1][j-1] == X[i-1][j-1] + 2*X[i-1][j] + X[i-1][j+1]
                                                -X[i+1][j-1] - 2*X[i+1][j] - X[i+1][j+1])
                model.addQConstr(filtered[i-1][j-1] * filtered[i-1][j-1] == ALPHA[i-1][j-1] * ALPHA[i-1][j-1]+ BETA[i-1][j-1] * BETA[i-1][j-1])

        # Constraint on X value
        for i in range(length_image+2):
            model.addLConstr(X[i][0] == 0)
            model.addLConstr(X[i][width_image+1] == 0)
        for j in range(width_image+2):
            model.addLConstr(X[0][j] == 0)
            model.addLConstr(X[length_image+1][j] == 0)

        # Intermediary constraint
        for i in range(length_image):
            for j in range(width_image):
                model.addLConstr(X_IA[i][j] == X[i+1][j+1]-Ia[i][j])
                model.addConstr(ABS[i][j] == gp.abs_(X_IA[i][j]))

        # Optimize model

        # print("\n Time before optimize : ",time()-start)
        before = time()
        model.optimize()
        # print("\n Time after optimize : ",time()-start)
        # print("\n Model Status :",model.status)
        vars = model.getVars()
        res = None
        # print(vars)
        # print(model.status)
        # print(model.SolCount)
        sol_count = model.SolCount
        objective_value = None
        if sol_count > 0:
            res = []
            for i in range(1,length_image+1):
                temp = []
                for j in range(1,width_image+1):
                    temp.append(float(vars[i*(width_image+2)+j].X))
                res.append(temp)
            objective_value = model.ObjVal
        if model.status != 2:
            if model.status == 3: # infeasable
                # print("infeasable")
                return [-2, time()-before, None, None]
            else:
                # print("time out")
                return [-3, time()-before, objective_value, res]
        # print("optimal")
        return [1, time()-before, objective_value, res]
        # for i, v in enumerate(model.getVars()):
        #     print(' %s %g \t' % (v.VarName, v.X),end="")
        return res
        print("***********************************************\n Model Status\n")
        print(model.status)
        if(model.status == 2):
            print('\n Obj: %g\n' % model.ObjVal)
        return(model.status)
        print()
        for i, v in enumerate(model.getVars()):
            print(' %s %g \t' % (v.VarName, v.X),end="")
            #print()
        vars = model.getVars()

        print('\n Obj: %g\n' % model.ObjVal)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

def max_filtered_value_model():
    """!
    max_filtered_value_model function to find the born value of max_filtered_value for the quadratic model.
    """

    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    start = time()
    # Create a new model
    model = gp.Model("mip1", env=env)
    model.Params.NonConvex = 2

    length_image = 1
    width_image = 1
    X = model.addMVar((length_image+2, width_image+2), name="X", lb=0, ub=255, vtype=GRB.INTEGER)
    # Variable for intermediate constraint
    ALPHA = model.addMVar((length_image, width_image), name="ALPHA",lb=-1200, ub=1200, vtype=GRB.CONTINUOUS)
    BETA = model.addMVar((length_image, width_image), name="BETA",lb=-1200, ub=1200, vtype=GRB.CONTINUOUS)
    filtered = model.addMVar((length_image, width_image), name="Fa", lb=0, vtype=GRB.CONTINUOUS)

    model.setObjective(sum(filtered[i][j] for i in range(length_image) for j in range(width_image)), GRB.MAXIMIZE)

    for i in range(1,length_image+1):
        for j in range(1,width_image+1):
            model.addLConstr(ALPHA[i-1][j-1] == X[i-1][j-1] + 2*X[i][j-1] + X[i+1][j-1]
                                                -X[i-1][j+1] - 2*X[i][j+1] - X[i+1][j+1])
            model.addLConstr(BETA[i-1][j-1] == X[i-1][j-1] + 2*X[i-1][j] + X[i-1][j+1]
                                                -X[i+1][j-1] - 2*X[i+1][j] - X[i+1][j+1])
            model.addQConstr(filtered[i-1][j-1] * filtered[i-1][j-1] == ALPHA[i-1][j-1] * ALPHA[i-1][j-1]+ BETA[i-1][j-1] * BETA[i-1][j-1])

    model.optimize()
    print(model.ObjVal)
