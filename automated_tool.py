
import matplotlib.pyplot as plt
from sklearn import linear_model


def pareto(df, cols, inputs):    
    arr = cols.tolist()
    sorted_df = df.sort_values(by=[cols[0]], ignore_index=True)
    temp=[]
    dic = {}
    x = sorted_df[cols[0:inputs]] #Setting the independent variables
    temp = arr[0:inputs]

    for i in range(inputs,len(arr)):
        y = sorted_df[arr[i]]
        var = arr[i]
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        dic[var]= regr.coef_

        coeff=[abs(i) for i in dic[var]]
        c_nor=[]
        c_abs=[]
        for i in range(len(dic[var])):
            if dic[var][i]>0:
                c_nor.append(coeff[i])
                c_abs.append(0)
            else:
                c_nor.append(0)
                c_abs.append(coeff[i])

        plt.figure()
        plt.bar(temp,c_nor,width = 0.9,color='blue',label='positive')
        plt.bar(temp,c_abs,width = 0.9,color ='red',label='negative')
        plt.xticks(rotation=45)
        plt.title('Pareto Representation of '+ var)
        plt.ylabel(var)
        plt.legend()
        plt.show()

""" MULTI OBJECTIVE GA and Single Objective GA
The target is to maximize or minimize the cubic equations:
    y = ax^3+bx^2+cx+d
    where a,b,c,d varies with every parameter
    
    What are the best values for the 1 unknown x?
    We are going to use the genetic algorithm for the best possible values after a number of generations.
"""
import pandas as pd
import numpy as np
import time
import sys
from mpl_toolkits import mplot3d

def cal_pop_fitness(equation_inputs, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
    fitness = pop**3*equation_inputs[0]+pop**2*equation_inputs[1]+pop*equation_inputs[2]+equation_inputs[3]
    return fitness

def select_mating_pool(pop, fitness, num_parents, choice):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        if choice == 'max':
            max_fitness_idx = np.where(fitness == np.max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num, :] = pop[max_fitness_idx, :]
            fitness[max_fitness_idx] = -99999999999 # Fitness value is set to a very small no. to avoid reselection of this solution
        elif choice == 'min':
            min_fitness_idx = np.where(fitness == np.min(fitness))
            min_fitness_idx = min_fitness_idx[0][0]
            parents[parent_num, :] = pop[min_fitness_idx, :]
            fitness[min_fitness_idx] = 99999999999 # Fitness value is set to a very large no. to avoid reselection of this solution
        else:
            sys.exit("Invalid choice!")
    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually it is at the center.
    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, mean):
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        random_value = np.random.uniform(-mean/100000000, mean/100000000, 1)
        offspring_crossover[idx, 0] = offspring_crossover[idx, 0] + random_value
    return offspring_crossover


def genetic(num_inputs):
    # Number of the weights we are looking to optimize.
    num_weights = 1
    print("-"*60+"\n"+"Welcome to the Genetic Algorithm (GA) Tool\n"+"-"*60+"\n")
    print("1) Single Objective GA\n2) Multi Objective GA\n")
    opt = int(input("Please select n option number from above (just digits): "))
    if opt!=1 and opt!=2:
        sys.exit("Invalid choice!")
    else:
        filename = str(input('\nPlease enter the name of the input csv file: '))  
        print()
        if (filename[-3:]!='csv'):
            filename = filename +'.csv'
        dataframe = pd.read_csv(filename)
        columns = np.array(dataframe.keys())
        
        sol_per_pop = len(dataframe)
        num_parents_mating = int(len(dataframe)/2) 
    
    
        # Defining the population size.
        pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
    
        num_generations = int(input('\nPlease enter the desired number of iterations (just digits): '))
        print()
        #num_generations = 25
    
        total_time = int(input('\nPlease enter the desired running duration of code in seconds (just digits): '))
        print()
        time_per_input = total_time/num_inputs
        time_period = time_per_input/(len(columns)-num_inputs)
    
        best_vals = {}
        best_fit_vals = {}
    
        choice = {}
        if opt == 1:
            limit = num_inputs+1 
        elif opt == 2: 
            limit = len(columns)
        print("Type 'min' to minimize a parameter or type 'max' to maximize it:\n")
        for col in range(num_inputs, limit):
            choice[columns[col]] = input(columns[col]+' : ')
        
        for inp in range(num_inputs):
            best_vals[columns[inp]] = {}
            best_fit_vals[columns[inp]] = {}
            for col in range(num_inputs, limit): # Stroke will be initial pop everytime, only eq inputs will change
                print('\n\tWorking for {} w.r.t. {}'.format(columns[col], columns[inp]))
    
                start = time.time()
                #Creating the initial population.    
                X = (np.array(dataframe[columns[inp]]))
                new_population = X.reshape(len(X), 1)
                #print('initial population: ', new_population) # x
                X_mean = np.mean(X)
    
                y = (np.array(dataframe[columns[col]]))
    
                # Inputs of the equation.
                eq_coeffs = np.poly1d(np.polyfit(X, y, 3))
                equation_inputs = np.array(eq_coeffs)
                #print('equation_inputs: ',equation_inputs)
    
                # Measing the fitness of each chromosome in the initial population.
                fitness = cal_pop_fitness(equation_inputs, new_population)
    
                a, b, c, d = np.polyfit(X, y, 3)
                title =  columns[inp] + ' vs ' + columns[col]
                plt.title(title)
                plt.xlabel(columns[inp])
                plt.ylabel(columns[col]) 
                #plt.plot(X, a*X*X*X+b*X*X+c*X+d, label ='Fitted Curve', color='blue')
                plt.scatter(X, y, label ='Original Data', color='red')
    
                for generation in range(num_generations):
                   # print("Generation : ", generation+1)
    
                    # Selecting the best parents in the population for mating.
                    parents = select_mating_pool(new_population, fitness, 
                                                    num_parents_mating, choice[columns[col]])
    
                    # Generating next generation using crossover.
                    offspring_crossover = crossover(parents,
                                                      offspring_size=(pop_size[0]-parents.shape[0], num_weights))
    
                    # Adding some variations to the offsrping using mutation.
                    offspring_mutation = mutation(offspring_crossover, X_mean)
    
                    # Creating the new population based on the parents and offspring.
                    new_population[0:parents.shape[0], :] = parents
                    new_population[parents.shape[0]:, :] = offspring_mutation
                    #print('new_population: ', new_population)
    
                    # Measing the fitness of each chromosome in the new population.
                    fitness = cal_pop_fitness(equation_inputs, new_population)
    
                    # The best result in the current iteration.
                    if choice[columns[col]] == 'max':
                        best_fit = np.max(fitness)
                    elif choice[columns[col]] == 'min':
                        best_fit = np.min(fitness)
                    else:
                        sys.exit("Invalid choice!")
    
                    best_index = np.where(fitness == best_fit)
                    best_val = new_population[best_index]
                    #print("{} : {} \n{} : {} \n".format(columns[col], best_fit, columns[inp], best_val))
    
                    if time.time() > start + time_period : 
                        break
    
    
                # Getting the best solution after iterating finishing all generations.
                best_match_idx = np.where(fitness == best_fit)
    
                best = new_population[best_match_idx]
                best_vals[columns[inp]][columns[col]] = best
                best_fit_vals[columns[inp]][columns[col]] = best_fit
    
                print("\nBest solution of {} w.r.t. {} : {}".format(columns[inp], columns[col], best))
                #print("Best solution of {} : {} \n".format(columns[col], fitness[best_match_idx]))
                print("Best solution of {} : {} \n".format(columns[col], best_fit))
    
                plt.plot(best_vals[columns[inp]][columns[col]], fitness[best_match_idx], label ='Optimum Result', marker="o", markersize=12, markeredgecolor="black", markerfacecolor="green")
    
                #add legend to plot
                plt.legend() 
                plt.grid()
                plt.show()
          
            for col in range(num_inputs, limit):    
                plt.scatter(best_vals[columns[inp]][columns[col]][0], best_fit_vals[columns[inp]][columns[col]], label ='Optimum', marker="o", color="blue")
                plt.title("{} vs {}".format(columns[inp], columns[col]))
                plt.xlabel(columns[inp])
                plt.ylabel(columns[col])
                plt.legend()
                plt.grid()
                plt.show()
              
        for inp in range(num_inputs):
            print('\nBest values of {} w.r.t. every parameter are as follows:'.format(columns[inp]))
            for col in range(num_inputs, limit):
                print('For {} {} = {}, {} = {}'.format(choice[columns[col]], columns[col], best_fit_vals[columns[inp]][columns[col]], columns[inp], best_vals[columns[inp]][columns[col]][0]))
    
        if limit-num_inputs == 3:
            fig = plt.figure(figsize = (8, 8))
            ax = plt.axes(projection = '3d')
    
            ax.scatter(best_fit_vals[columns[inp]][columns[num_inputs]], best_fit_vals[columns[inp]][columns[num_inputs+1]], best_fit_vals[columns[inp]][columns[num_inputs+2]], c='Blue')
            ax.set_title('3D plotting of optimum values')
            ax.set_xlabel(columns[num_inputs], fontsize=10)
            ax.set_ylabel(columns[num_inputs+1], fontsize=10)
            ax.set_zlabel(columns[num_inputs+2], fontsize=10)
            plt.grid()
            plt.show()
        
        # Pareto plotting
        pareto(dataframe, columns, num_inputs)
    
# ===========================================================================================
# Function for accepting the user input (choice) for desired DOE and the input CSV file name
# ===========================================================================================

def user_input():
    print("-"*60+"\n"+"Welcome to the DoE Menu\n"+"-"*60+"\n")
    list_doe = ["1) Latin hypercube (simple)",
                "2) Latin hypercube (space-filling)",
                "3) Full factorial",
                "4) 2-level fractional factorial",
                "5) Plackett-Burman",
                "6) Sukharev Grid",
                "7) Box-Behnken",
                "8) Box-Wilson (Central-composite) with center-faced option",
                "9) Box-Wilson (Central-composite) with center-inscribed option",
                "10) Box-Wilson (Central-composite) with center-circumscribed option",
                "11) Random k-means cluster",
                "12) Maximin reconstruction",
                "13) Halton sequence based",
                "14) Uniform random matrix"
                ]
    
    for choice in list_doe:
        print(choice)
    print("-"*60)
    
    doe_choice = int(input("Please select n option number from above: "))
    print()
    infile = str(input("Please enter the name of the input csv file: "))
    print()
    
    if (infile[-3:]!='csv'):
        infile=infile+'.csv'
      
    return (doe_choice,infile)
#====================
# Essential imports
#====================
from pyDOE import *
from diversipy import *
import pandas as pd
import numpy as np

# ===========================================================================================================
# Function for constructing a DataFrame from a numpy array generated by PyDOE function and individual lists
# ===========================================================================================================

def construct_df(x,r):
    df=pd.DataFrame(data=x,dtype='float32')
    for i in df.index:
        for j in range(len(list(df.iloc[i]))):
            df.iloc[i][j]=r[j][int(df.iloc[i][j])]
    return df

# ===================================================================================================
# Function for constructing a DataFrame from a matrix with floating point numbers between -1 and +1
# ===================================================================================================

def construct_df_from_matrix(x,factor_array):
    """
    This function constructs a DataFrame out of x and factor_array, both of which are assumed to be numpy arrays.
    It projects the numbers in the x (which is output of a design-of-experiment build) to the factor array ranges.
    Here factor_array is assumed to have only min and max ranges.
    Matrix x is assumed to have numbers ranging from -1 to 1.
    """
    
    row_num=x.shape[0] # Number of rows in the matrix x
    col_num=x.shape[1] # Number of columns in the matrix x
    
    empty=np.zeros((row_num,col_num))  
    
    def simple_substitution(idx,factor_list):
        if idx==-1:
            return factor_list[0]
        elif idx==0:
            return factor_list[1]
        elif idx==1:
            return factor_list[2]
        else:
            alpha=np.abs(factor_list[2]-factor_list[0])/2
            if idx<0:
                beta=np.abs(idx)-1
                return factor_list[0]-(beta*alpha)
            else:
                beta=idx-1
                return factor_list[2]+(beta*alpha)
        
    for i in range(row_num):
        for j in range(col_num):
            empty[i,j] = simple_substitution(x[i,j],factor_array[j])
        
    return pd.DataFrame(data=empty)

# =================================================================================================
# Function for constructing a DataFrame from a matrix with floating point numbers between 0 and 1
# =================================================================================================

def construct_df_from_random_matrix(x,factor_array):
    """
    This function constructs a DataFrame out of matrix x and factor_array, both of which are assumed to be numpy arrays.
    It projects the numbers in the x (which is output of a design-of-experiment build) to the factor array ranges.
    Here factor_array is assumed to have only min and max ranges.
    Matrix x is assumed to have numbers ranging from 0 to 1 only.
    """
    
    row_num=x.shape[0] # Number of rows in the matrix x
    col_num=x.shape[1] # Number of columns in the matrix x
    
    empty=np.zeros((row_num,col_num))  
    
    def simple_substitution(idx,factor_list):
        alpha=np.abs(factor_list[1]-factor_list[0])
        beta=idx
        return factor_list[0]+(beta*alpha)
        
    for i in range(row_num):
        for j in range(col_num):
            empty[i,j] = simple_substitution(x[i,j],factor_array[j])
        
    return pd.DataFrame(data=empty)

# ======================================================================================
# Function for building full factorial DataFrame from a dictionary of process variables
# ======================================================================================

def build_full_fact(factor_level_ranges):
    """
    Builds a full factorial design dataframe from a dictionary of factor/level ranges
    Example of the process variable dictionary:
    {'Pressure':[50,60,70],'Temperature':[290, 320, 350],'Flow rate':[0.9,1.0]}
    """
    
    factor_lvl_count=[]
    factor_lists=[]
    
    for key in factor_level_ranges:
        factor_lvl_count.append(len(factor_level_ranges[key]))
        factor_lists.append(factor_level_ranges[key])
    
    x = fullfact_corrected(factor_lvl_count)
    df=construct_df(x,factor_lists)
    df.columns=factor_level_ranges.keys()
    
    return df

# ==================================================================================================================================================
# Function for building 2-level fractional factorial DataFrame from a dictionary and a generator string
# ================================================================================================================================================================

def build_frac_fact(factor_level_ranges,gen_string):
    """
    Builds a full factorial design dataframe from a dictionary of factor/level ranges.
    Only min and max values of the range are required.
    Example of the dictionary:
    {'Pressure':[50,70],'Temperature':[290, 350],'Flow rate':[0.9,1.0]}
    
    This function requires a little more knowledge of how the confounding will be allowed. 
    This means that some factor effects get muddled with other interaction effects, so it’s harder to distinguish between them).
    
    Let’s assume that we just can’t afford (for whatever reason) the number of runs in a full-factorial design. We can systematically decide on a fraction of the full-factorial by allowing some of the factor main effects to be confounded with other factor interaction effects. 
    This is done by defining an alias structure that defines, symbolically, these interactions. These alias structures are written like “C = AB” or “I = ABC”, or “AB = CD”, etc. 
    These define how one column is related to the others.
    
    EXAMPLE
    ------------
    For example, the alias “C = AB” or “I = ABC” indicate that there are three factors (A, B, and C) and that the main effect of factor C is confounded with the interaction effect of the product AB, and by extension, A is confounded with BC and B is confounded with AC. 
    A full- factorial design with these three factors results in a design matrix with 8 runs, but we will assume that we can only afford 4 of those runs. 
    To create this fractional design, we need a matrix with three columns, one for A, B, and C, only now where the levels in the C column is created by the product of the A and B columns.
    """
    
    factor_count=len(factor_level_ranges)
    factor_lists=[]
    
    for key in factor_level_ranges:
        if len(factor_level_ranges[key])!=2:
            factor_level_ranges[key][1]=factor_level_ranges[key][-1]
            factor_level_ranges[key]=factor_level_ranges[key][:2]
            print(f"{key} had more than two levels. Assigning the end point to the high level.")
    
    if factor_count!=len(gen_string.split(' ')):
        print("Length of the generator string for the fractional factorial build does not match the length of the process variables dictionary")
        return None
    
    for key in factor_level_ranges:
        factor_lists.append(factor_level_ranges[key])
    
    x = fracfact(gen_string)
    
    def index_change(x):
        if x==-1:
            return 0
        else:
            return x
    vfunc=np.vectorize(index_change)
    x=vfunc(x)
       
    df=construct_df(x,factor_lists)
    df.columns=factor_level_ranges.keys()
    
    return df

# =====================================================================================
# Function for building Plackett-Burman designs from a dictionary of process variables
# =====================================================================================

def build_plackett_burman(factor_level_ranges):
    """
    Builds a Plackett-Burman dataframe from a dictionary of factor/level ranges.
    Only min and max values of the range are required.
    Example of the dictionary:
    {'Pressure':[50,70],'Temperature':[290, 350],'Flow rate':[0.9,1.0]}
    
    Plackett–Burman designs are experimental designs presented in 1946 by Robin L. Plackett and J. P. Burman while working in the British Ministry of Supply.(Their goal was to find experimental designs for investigating the dependence of some measured quantity on a number of independent variables (factors), each taking L levels, in such a way as to minimize the variance of the estimates of these dependencies using a limited number of experiments. 
    
    Interactions between the factors were considered negligible. The solution to this problem is to find an experimental design where each combination of levels for any pair of factors appears the same number of times, throughout all the experimental runs (refer to table). 
    A complete factorial design would satisfy this criterion, but the idea was to find smaller designs.
    
    These designs are unique in that the number of trial conditions (rows) expands by multiples of four (e.g. 4, 8, 12, etc.). 
    The max number of columns allowed before a design increases the number of rows is always one less than the next higher multiple of four.
    """
    
    for key in factor_level_ranges:
        if len(factor_level_ranges[key])!=2:
            factor_level_ranges[key][1]=factor_level_ranges[key][-1]
            factor_level_ranges[key]=factor_level_ranges[key][:2]
            print(f"{key} had more than two levels. Assigning the end point to the high level.")
    
    factor_count=len(factor_level_ranges)
    factor_lists=[]
    
    for key in factor_level_ranges:
        factor_lists.append(factor_level_ranges[key])
    
    x = pbdesign(factor_count)
    
    def index_change(x):
        if x==-1:
            return 0
        else:
            return x
    vfunc=np.vectorize(index_change)
    x=vfunc(x)
       
    df=construct_df(x,factor_lists)
    df.columns=factor_level_ranges.keys()
    
    return df

# ===================================================================================
# Function for building Sukharev Grid designs from a dictionary of process variables
# ===================================================================================

def build_sukharev(factor_level_ranges,num_samples=None):
    """
    Builds a Sukharev-grid hypercube design dataframe from a dictionary of factor/level ranges.
    Number of samples raised to the power of (1/dimension), where dimension is the number of variables, must be an integer.
    Only min and max values of the range are required.
    Example of the dictionary:
    {'Pressure':[50,70],'Temperature':[290, 350],'Flow rate':[0.9,1.0]}
    num_samples: Number of samples to be generated
    
    Special property of this grid is that points are not placed on the boundaries of the hypercube, but at centroids of the  subcells constituted by individual samples. 
    This design offers optimal results for the covering radius regarding distances based on the max-norm.
    """
    for key in factor_level_ranges:
        if len(factor_level_ranges[key])!=2:
            factor_level_ranges[key][1]=factor_level_ranges[key][-1]
            factor_level_ranges[key]=factor_level_ranges[key][:2]
            print(f"{key} had more than two levels. Assigning the end point to the high level.")
    
    factor_count=len(factor_level_ranges)
    factor_lists=[]
    
    for key in factor_level_ranges:
        factor_lists.append(factor_level_ranges[key])
    
    check=num_samples**((1/factor_count))
    if (check-int(check)>1e-5):
        num_samples=(int(check)+1)**(factor_count)
        print("\nNumber of samples not adequate to fill a Sukharev grid. Increasing sample size to: ",num_samples)
    
    x = sukharev_grid(num_points=num_samples,dimension=factor_count)
    factor_lists=np.array(factor_lists)
    
    df = construct_df_from_random_matrix(x,factor_lists)
    df.columns=factor_level_ranges.keys()
    return df

# ===================================================================================
# Function for building Box-Behnken designs from a dictionary of process variables
# ===================================================================================

def build_box_behnken(factor_level_ranges,center=1):
    """
    Builds a Box-Behnken design dataframe from a dictionary of factor/level ranges.
    Note 3 levels of factors are necessary. If not given, the function will automatically create 3 levels by linear mid-section method.
    Example of the dictionary:
    {'Pressure':[50,60,70],'Temperature':[290, 320, 350],'Flow rate':[0.9,1.0,1.1]}
    
    In statistics, Box–Behnken designs are experimental designs for response surface methodology, devised by George E. P. Box and Donald Behnken in 1960, to achieve the following goals:
        * Each factor, or independent variable, is placed at one of three equally spaced values, usually coded as −1, 0, +1. (At least three levels are needed for the following goal.)
        * The design should be sufficient to fit a quadratic model, that is, one containing squared terms, products of two factors, linear terms and an intercept.
        * The ratio of the number of experimental points to the number of coefficients in the quadratic model should be reasonable (in fact, their designs kept it in the range of 1.5 to 2.6).*estimation variance should more or less depend only on the distance from the centre (this is achieved exactly for the designs with 4 and 7 factors), and should not vary too much inside the smallest (hyper)cube containing the experimental points.
    """
    for key in factor_level_ranges:
        if len(factor_level_ranges[key])==2:
            factor_level_ranges[key].append((factor_level_ranges[key][0]+factor_level_ranges[key][1])/2)
            factor_level_ranges[key].sort()
            print(f"{key} had only two end points. Creating a mid-point by averaging them")
    
    factor_count=len(factor_level_ranges)
    factor_lists=[]
    
    for key in factor_level_ranges:
        factor_lists.append(factor_level_ranges[key])
    
    x = bbdesign_corrected(factor_count,center=center)
    x=x+1 #Adjusting the index up by 1

    df=construct_df(x,factor_lists)
    df.columns=factor_level_ranges.keys()
    
    return df

# =====================================================================================================
# Function for building central-composite (Box-Wilson) designs from a dictionary of process variables
# ===================================================================================================== 

def build_central_composite(factor_level_ranges,center=(2,2),alpha='o',face='ccc'):
    """
    Builds a central-composite design dataframe from a dictionary of factor/level ranges.
    Only min and max values of the range are required.
    Example of the dictionary:
    {'Pressure':[50,70],'Temperature':[290, 350],'Flow rate':[0.9,1.0]}
    
    In statistics, a central composite design is an experimental design, useful in response surface methodology, for building a second order (quadratic) model for the response variable without needing to use a complete three-level factorial experiment.
    The design consists of three distinct sets of experimental runs:
        * A factorial (perhaps fractional) design in the factors studied, each having two levels;
        * A set of center points, experimental runs whose values of each factor are the medians of the values used in the factorial portion. This point is often replicated in order to improve the precision of the experiment;
        * A set of axial points, experimental runs identical to the centre points except for one factor, which will take on values both below and above the median of the two factorial levels, and typically both outside their range. All factors are varied in this way.
    """
    for key in factor_level_ranges:
        if len(factor_level_ranges[key])!=2:
            factor_level_ranges[key][1]=factor_level_ranges[key][-1]
            factor_level_ranges[key]=factor_level_ranges[key][:2]
            print(f"{key} had more than two levels. Assigning the end point to the high level.")
    
    
    # Creates the mid-points by averaging the low and high levels
    for key in factor_level_ranges:
        if len(factor_level_ranges[key])==2:
            factor_level_ranges[key].append((factor_level_ranges[key][0]+factor_level_ranges[key][1])/2)
            factor_level_ranges[key].sort()
    
    factor_count=len(factor_level_ranges)
    factor_lists=[]
    
    for key in factor_level_ranges:
        factor_lists.append(factor_level_ranges[key])
    
    x = ccdesign(factor_count,center=center,alpha=alpha,face=face)
    factor_lists=np.array(factor_lists)
    
    df = construct_df_from_matrix(x,factor_lists)
    df.columns=factor_level_ranges.keys()
    return df

# ====================================================================================
# Function for building simple Latin Hypercube from a dictionary of process variables
# ====================================================================================

def build_lhs(factor_level_ranges, num_samples=None, prob_distribution=None):
    """
    Builds a Latin Hypercube design dataframe from a dictionary of factor/level ranges.
    Only min and max values of the range are required.
    Example of the dictionary:
    {'Pressure':[50,70],'Temperature':[290, 350],'Flow rate':[0.9,1.0]}
    num_samples: Number of samples to be generated
    prob_distribution: Analytical probability distribution to be applied over the randomized sampling. 
    Takes strings like: 'Normal', 'Poisson', 'Exponential', 'Beta', 'Gamma'

    Latin hypercube sampling (LHS) is a form of stratified sampling that can be applied to multiple variables. The method commonly used to reduce the number or runs necessary for a Monte Carlo simulation to achieve a reasonably accurate random distribution. LHS can be incorporated into an existing Monte Carlo model fairly easily, and work with variables following any analytical probability distribution.
    """
    for key in factor_level_ranges:
        if len(factor_level_ranges[key])!=2:
            factor_level_ranges[key][1]=factor_level_ranges[key][-1]
            factor_level_ranges[key]=factor_level_ranges[key][:2]
            #print(f"{key} had more than two levels. Assigning the end point to the high level.")
    
    factor_count=len(factor_level_ranges)
    factor_lists=[]
    
    if num_samples==None:
        num_samples=factor_count
    
    for key in factor_level_ranges:
        factor_lists.append(factor_level_ranges[key])
    
    x = lhs(n=factor_count,samples=num_samples)
    factor_lists=np.array(factor_lists)
    
    df = construct_df_from_random_matrix(x,factor_lists)
    df.columns=factor_level_ranges.keys()
    return df

# ============================================================================================
# Function for building space-filling Latin Hypercube from a dictionary of process variables
# ============================================================================================

def build_space_filling_lhs(factor_level_ranges, num_samples=None):
    """
    Builds a space-filling Latin Hypercube design dataframe from a dictionary of factor/level ranges.
    Only min and max values of the range are required.
    Example of the dictionary:
    {'Pressure':[50,70],'Temperature':[290, 350],'Flow rate':[0.9,1.0]}
    num_samples: Number of samples to be generated
    """
    for key in factor_level_ranges:
        if len(factor_level_ranges[key])!=2:
            factor_level_ranges[key][1]=factor_level_ranges[key][-1]
            factor_level_ranges[key]=factor_level_ranges[key][:2]
            #print(f"{key} had more than two levels. Assigning the end point to the high level.")
    
    factor_count=len(factor_level_ranges)
    factor_lists=[]
    
    if num_samples==None:
        num_samples=factor_count
    
    for key in factor_level_ranges:
        factor_lists.append(factor_level_ranges[key])
    
    x = transform_spread_out(lhd_matrix(num_points=num_samples,dimension=factor_count)) # create latin hypercube design
    factor_lists=np.array(factor_lists)
     
    df = construct_df_from_random_matrix(x,factor_lists)
    df.columns=factor_level_ranges.keys()
    return df

# =====================================================================================================
# Function for building designs with random _k-means_ clusters from a dictionary of process variables
# =====================================================================================================

def build_random_k_means(factor_level_ranges, num_samples=None):
    """
    This function aims to produce a centroidal Voronoi tesselation of the unit random hypercube and generate k-means clusters.
    Only min and max values of the range are required.
    Example of the dictionary:
    {'Pressure':[50,70],'Temperature':[290, 350],'Flow rate':[0.9,1.0]}
    num_samples: Number of samples to be generated
    """
    for key in factor_level_ranges:
        if len(factor_level_ranges[key])!=2:
            factor_level_ranges[key][1]=factor_level_ranges[key][-1]
            factor_level_ranges[key]=factor_level_ranges[key][:2]
            print(f"{key} had more than two levels. Assigning the end point to the high level.")
    
    factor_count=len(factor_level_ranges)
    factor_lists=[]
    
    if num_samples==None:
        num_samples=factor_count
    
    for key in factor_level_ranges:
        factor_lists.append(factor_level_ranges[key])
    
    x = random_k_means(num_points=num_samples,dimension=factor_count) # create latin hypercube design
    factor_lists=np.array(factor_lists)
    
    df = construct_df_from_random_matrix(x,factor_lists)
    df.columns=factor_level_ranges.keys()
    return df

# =============================================================================================
# Function for building maximin reconstruction matrix from a dictionary of process variables
# =============================================================================================

def build_maximin(factor_level_ranges, num_samples=None):
    """
    Builds a maximin reconstructed design dataframe from a dictionary of factor/level ranges.
    Only min and max values of the range are required.
    Example of the dictionary:
    {'Pressure':[50,70],'Temperature':[290, 350],'Flow rate':[0.9,1.0]}
    num_samples: Number of samples to be generated
    
    This algorithm carries out a user-specified number of iterations to maximize the minimal distance of a point in the set to 
        * other points in the set, 
        * existing (fixed) points, 
        * the boundary of the hypercube.
    """
    for key in factor_level_ranges:
        if len(factor_level_ranges[key])!=2:
            factor_level_ranges[key][1]=factor_level_ranges[key][-1]
            factor_level_ranges[key]=factor_level_ranges[key][:2]
            print(f"{key} had more than two levels. Assigning the end point to the high level.")
    
    factor_count=len(factor_level_ranges)
    factor_lists=[]
    
    if num_samples==None:
        num_samples=factor_count
    
    for key in factor_level_ranges:
        factor_lists.append(factor_level_ranges[key])
    
    x = maximin_reconstruction(num_points=num_samples,dimension=factor_count) # create latin hypercube design
    factor_lists=np.array(factor_lists)
      
    df = construct_df_from_random_matrix(x,factor_lists)
    df.columns=factor_level_ranges.keys()
    return df

# ========================================================================================
# Function for building Halton matrix based design from a dictionary of process variables
# ========================================================================================

def build_halton(factor_level_ranges, num_samples=None):
    """
    Builds a quasirandom dataframe from a dictionary of factor/level ranges using prime numbers as seed.
    Only min and max values of the range are required.
    Example of the dictionary:
    {'Pressure':[50,70],'Temperature':[290, 350],'Flow rate':[0.9,1.0]}
    num_samples: Number of samples to be generated

    Quasirandom sequence using the default initialization with first n prime numbers equal to the number of factors/variables.
    """
    for key in factor_level_ranges:
        if len(factor_level_ranges[key])!=2:
            factor_level_ranges[key][1]=factor_level_ranges[key][-1]
            factor_level_ranges[key]=factor_level_ranges[key][:2]
            print(f"{key} had more than two levels. Assigning the end point to the high level.")
    
    factor_count=len(factor_level_ranges)
    factor_lists=[]
    
    if num_samples==None:
        num_samples=factor_count
    
    for key in factor_level_ranges:
        factor_lists.append(factor_level_ranges[key])
    
    x = halton(num_points=num_samples,dimension=factor_count) # create Halton matrix design
    factor_lists=np.array(factor_lists)
    
    df = construct_df_from_random_matrix(x,factor_lists)
    df.columns=factor_level_ranges.keys()
    return df

# ==========================================================================================
# Function for building uniform random design matrix from a dictionary of process variables
# ==========================================================================================

def build_uniform_random (factor_level_ranges, num_samples=None):
    """
    Builds a design dataframe with samples drawn from uniform random distribution based on a dictionary of factor/level ranges.
    Only min and max values of the range are required.
    Example of the dictionary:
    {'Pressure':[50,70],'Temperature':[290, 350],'Flow rate':[0.9,1.0]}
    num_samples: Number of samples to be generated
    """
    for key in factor_level_ranges:
        if len(factor_level_ranges[key])!=2:
            factor_level_ranges[key][1]=factor_level_ranges[key][-1]
            factor_level_ranges[key]=factor_level_ranges[key][:2]
            print(f"{key} had more than two levels. Assigning the end point to the high level.")
    
    factor_count=len(factor_level_ranges)
    factor_lists=[]
    
    if num_samples==None:
        num_samples=factor_count
    
    for key in factor_level_ranges:
        factor_lists.append(factor_level_ranges[key])
    
    x = random_uniform(num_points=num_samples,dimension=factor_count) # create Halton matrix design
    factor_lists=np.array(factor_lists)
     
    df = construct_df_from_random_matrix(x,factor_lists)
    df.columns=factor_level_ranges.keys()
    return df



# ====================================================================
# Function to generate the DOE based on user's choice and input file
# ====================================================================

def generate_DOE(doe_choice, infile):
    """
    Generates the output design-of-experiment matrix by calling the appropriate function from the "DOE_function.py file".
    Returns the generated DataFrame (Pandas) and a filename (string) corresponding to the type of the DOE sought by the user. This filename string is used by the CSV writer function to write to the disk i.e. save the generated DataFrame in a CSV format.
    """
    
    dict_vars = read_variables_csv(infile)
    if type(dict_vars)!=int:
        factor_count=len(dict_vars)
    else:
        return (-1,-1)
    
    if doe_choice==1:
        num_samples=int(input("Please enter the number of random sample points to generate: "))
        print()
        df=build_lhs(dict_vars,num_samples=num_samples)
        filename='Simple_Latin_Hypercube_design'
    
    elif doe_choice==2:
        num_samples=int(input("Please enter the number of random sample points to generate: "))
        print()
        df=build_space_filling_lhs(dict_vars,num_samples=num_samples)
        filename='Space_filling_Latin_Hypercube_design'
        
    elif doe_choice==3:
        df=build_full_fact(dict_vars)
        filename='Full_factorial_design'
    
    elif doe_choice==4:
        print("For this choice, you will be asked to enter a generator string expression. Please only use small letters e.g. 'a b c bc' for the string. Make sure to put a space in between every variable. Please note that the number of character blocks must be identical to the number of factors you have in your input file.\n")
        gen_string=str(input("Please enter the generator string for the fractional factorial build: "))
        print()
        if len(gen_string.split(' '))!=factor_count:
            print("Length of the generator string does not match the number of factors/variables. Sorry!")
            return (-1,-1)
        df=build_frac_fact(dict_vars,gen_string)
        filename='Fractional_factorial_design'
    
    elif doe_choice==5:
        df=build_plackett_burman(dict_vars)
        filename='Plackett_Burman_design'
    
    elif doe_choice==6:
        num_samples=int(input("Please enter the number of samples: "))
        print()
        df=build_sukharev(dict_vars,num_samples)
        filename='Sukharev_grid_design'
    
    elif doe_choice==7:
        num_center=int(input("Please enter the number of center points to be repeated (if more than one): "))
        print()
        df=build_box_behnken(dict_vars,num_center)
        filename='Box_Behnken_design'
    
    elif doe_choice==8:
        #num_center=int(input("Please enter the number of center points to be repeated (if more than one): "))
        print()
        df=build_central_composite(dict_vars,face='ccf')
        filename='Box_Wilson_face_centered_design'
    
    elif doe_choice==9:
        #num_center=int(input("Please enter the number of center points to be repeated (if more than one): "))
        print()
        df=build_central_composite(dict_vars,face='cci')
        filename='Box_Wilson_face_inscribed_design'
    
    elif doe_choice==10:
        #num_center=int(input("Please enter the number of center points to be repeated (if more than one): "))
        print()
        df=build_central_composite(dict_vars,face='ccc')
        filename='Box_Wilson_face_circumscribed_design'
    
    elif doe_choice==11:
        num_samples=int(input("Please enter the number of random sample points to generate: "))
        print()
        df=build_random_k_means(dict_vars,num_samples=num_samples)
        filename='Random_k_means_design'
    
    elif doe_choice==12:
        num_samples=int(input("Please enter the number of random sample points to generate: "))
        print()
        df=build_maximin(dict_vars,num_samples=num_samples)
        filename='Maximin_reconstruction_design'
    
    elif doe_choice==13:
        num_samples=int(input("Please enter the number of random sample points to generate: "))
        print()
        df=build_halton(dict_vars,num_samples=num_samples)
        filename='Halton_sequence_design'
    
    elif doe_choice==14:
        num_samples=int(input("Please enter the number of random sample points to generate: "))
        print()
        df=build_uniform_random(dict_vars,num_samples=num_samples)
        filename='Uniform_random_matrix_design'

    return (df,filename)
import csv

# ==========================================================
# Function for reading a CSV file into a dictionary format
# ==========================================================

def read_variables_csv(csvfile):
    """
    Builds a Python dictionary object from an input CSV file.
    Helper function to read a CSV file on the disk, where user stores the limits/ranges of the process variables.
    """
    dict_key={}
    try:
        with open(csvfile) as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            for field in fields:
                lst=[]
                with open(csvfile) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        lst.append(float(row[field]))
                dict_key[field]=lst
    
        return dict_key
    except:
        print("Error in reading the specified file from the disk. Please make sure it is in current directory.")
        return -1
        
# ===============================================================
# Function for writing the design matrix into an output CSV file
# ===============================================================

def write_csv(df,filename):
    """
    Writes a CSV file on to the disk from the internal Pandas DataFrame object i.e. the computed design matrix
    """
    try:
        filename=filename+'.csv'
        df.to_csv(filename, index=False)
    except:
        return -1

"""
This code was originally published by the following individuals for use with
Scilab:
    Copyright (C) 2012 - 2013 - Michael Baudin
    Copyright (C) 2012 - Maria Christopoulou
    Copyright (C) 2010 - 2011 - INRIA - Michael Baudin
    Copyright (C) 2009 - Yann Collette
    Copyright (C) 2009 - CEA - Jean-Marc Martinez
    
    website: forge.scilab.org/index.php/p/scidoe/sourcetree/master/macros

Much thanks goes to these individuals. It has been converted to Python by 
Abraham Lee.
"""

import re
import numpy as np

#__all__ = ['np', 'fullfact_corrected', 'ff2n', 'fracfact']

def fullfact_corrected(levels):
    """
    Create a general full-factorial design
    
    Parameters
    ----------
    levels : array-like
        An array of integers that indicate the number of levels of each input
        design factor.
    
    Returns
    -------
    mat : 2d-array
        The design matrix with coded levels 0 to k-1 for a k-level factor
    
    Example
    -------
    ::
    
        >>> fullfact([2, 4, 3])
        array([[ 0.,  0.,  0.],
               [ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 1.,  1.,  0.],
               [ 0.,  2.,  0.],
               [ 1.,  2.,  0.],
               [ 0.,  3.,  0.],
               [ 1.,  3.,  0.],
               [ 0.,  0.,  1.],
               [ 1.,  0.,  1.],
               [ 0.,  1.,  1.],
               [ 1.,  1.,  1.],
               [ 0.,  2.,  1.],
               [ 1.,  2.,  1.],
               [ 0.,  3.,  1.],
               [ 1.,  3.,  1.],
               [ 0.,  0.,  2.],
               [ 1.,  0.,  2.],
               [ 0.,  1.,  2.],
               [ 1.,  1.,  2.],
               [ 0.,  2.,  2.],
               [ 1.,  2.,  2.],
               [ 0.,  3.,  2.],
               [ 1.,  3.,  2.]])
               
    """
    n = len(levels)  # number of factors
    nb_lines = np.prod(levels)  # number of trial conditions
    H = np.zeros((nb_lines, n))
    
    level_repeat = 1
    range_repeat = np.prod(levels)
    for i in range(n):
        range_repeat //= levels[i]
        lvl = []
        for j in range(levels[i]):
            lvl += [j]*level_repeat
        rng = lvl*range_repeat
        level_repeat *= levels[i]
        H[:, i] = rng
     
    return H
    
################################################################################

def ff2n(n):
    """
    Create a 2-Level full-factorial design
    
    Parameters
    ----------
    n : int
        The number of factors in the design.
    
    Returns
    -------
    mat : 2d-array
        The design matrix with coded levels -1 and 1
    
    Example
    -------
    ::
    
        >>> ff2n(3)
        array([[-1., -1., -1.],
               [ 1., -1., -1.],
               [-1.,  1., -1.],
               [ 1.,  1., -1.],
               [-1., -1.,  1.],
               [ 1., -1.,  1.],
               [-1.,  1.,  1.],
               [ 1.,  1.,  1.]])
       
    """
    return 2*fullfact_corrected([2]*n) - 1

################################################################################

def fracfact(gen):
    """
    Create a 2-level fractional-factorial design with a generator string.
    
    Parameters
    ----------
    gen : str
        A string, consisting of lowercase, uppercase letters or operators "-"
        and "+", indicating the factors of the experiment
    
    Returns
    -------
    H : 2d-array
        A m-by-n matrix, the fractional factorial design. m is 2^k, where k
        is the number of letters in ``gen``, and n is the total number of
        entries in ``gen``.
    
    Notes
    -----
    In ``gen`` we define the main factors of the experiment and the factors
    whose levels are the products of the main factors. For example, if
    
        gen = "a b ab"
    
    then "a" and "b" are the main factors, while the 3rd factor is the product
    of the first two. If we input uppercase letters in ``gen``, we get the same
    result. We can also use the operators "+" and "-" in ``gen``.
    
    For example, if
    
        gen = "a b -ab"
    
    then the 3rd factor is the opposite of the product of "a" and "b".
    
    The output matrix includes the two level full factorial design, built by
    the main factors of ``gen``, and the products of the main factors. The
    columns of ``H`` follow the sequence of ``gen``.
    
    For example, if
    
        gen = "a b ab c"
    
    then columns H[:, 0], H[:, 1], and H[:, 3] include the two level full
    factorial design and H[:, 2] includes the products of the main factors.
    
    Examples
    --------
    ::
    
        >>> fracfact("a b ab")
        array([[-1., -1.,  1.],
               [ 1., -1., -1.],
               [-1.,  1., -1.],
               [ 1.,  1.,  1.]])
       
        >>> fracfact("A B AB")
        array([[-1., -1.,  1.],
               [ 1., -1., -1.],
               [-1.,  1., -1.],
               [ 1.,  1.,  1.]])
        
        >>> fracfact("a b -ab c +abc")
        array([[-1., -1., -1., -1., -1.],
               [ 1., -1.,  1., -1.,  1.],
               [-1.,  1.,  1., -1.,  1.],
               [ 1.,  1., -1., -1., -1.],
               [-1., -1., -1.,  1.,  1.],
               [ 1., -1.,  1.,  1., -1.],
               [-1.,  1.,  1.,  1., -1.],
               [ 1.,  1., -1.,  1.,  1.]])
       
    """
    # Recognize letters and combinations
    A = [item for item in re.split('\-?\s?\+?', gen) if item]  # remove empty strings
    C = [len(item) for item in A]
    
    # Indices of single letters (main factors)
    I = [i for i, item in enumerate(C) if item==1]
    
    # Indices of letter combinations (we need them to fill out H2 properly).
    J = [i for i, item in enumerate(C) if item!=1]
    
    # Check if there are "-" or "+" operators in gen
    U = [item for item in gen.split(' ') if item]  # remove empty strings
    
    # If R1 is either None or not, the result is not changed, since it is a
    # multiplication of 1.
    R1 = _grep(U, '+')
    R2 = _grep(U, '-')
    
    # Fill in design with two level factorial design
    H1 = ff2n(len(I))
    H = np.zeros((H1.shape[0], len(C)))
    H[:, I] = H1
    
    # Recognize combinations and fill in the rest of matrix H2 with the proper
    # products
    for k in J:
        # For lowercase letters
        xx = np.array([ord(c) for c in A[k]]) - 97
        
        # For uppercase letters
        if np.any(xx<0):
            xx = np.array([ord(c) for c in A[k]]) - 65
        
        H[:, k] = np.prod(H1[:, xx], axis=1)
    
    # Update design if gen includes "-" operator
    if R2:
        H[:, R2] *= -1
        
    # Return the fractional factorial design
    return H
    
def _grep(haystack, needle):
    try:
        haystack[0]
    except (TypeError, AttributeError):
        return [0] if needle in haystack else []
    else:
        locs = []
        for idx, item in enumerate(haystack):
            if needle in item:
                locs += [idx]
        return locs


#__all__ = ['bbdesign_corrected']

def bbdesign_corrected(n, center=None):
    """
    Create a Box-Behnken design
    
    Parameters
    ----------
    n : int
        The number of factors in the design
    
    Optional
    --------
    center : int
        The number of center points to include (default = 1).
    
    Returns
    -------
    mat : 2d-array
        The design matrix
    
    Example
    -------
    ::
    
        >>> bbdesign(3)
        array([[-1., -1.,  0.],
               [ 1., -1.,  0.],
               [-1.,  1.,  0.],
               [ 1.,  1.,  0.],
               [-1.,  0., -1.],
               [ 1.,  0., -1.],
               [-1.,  0.,  1.],
               [ 1.,  0.,  1.],
               [ 0., -1., -1.],
               [ 0.,  1., -1.],
               [ 0., -1.,  1.],
               [ 0.,  1.,  1.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.]])
        
    """
    assert n>=3, 'Number of variables must be at least 3'
    
    # First, compute a factorial DOE with 2 parameters
    H_fact = ff2n(2)
    # Now we populate the real DOE with this DOE
    
    # We made a factorial design on each pair of dimensions
    # - So, we created a factorial design with two factors
    # - Make two loops
    Index = 0
    nb_lines = int((0.5*n*(n-1))*H_fact.shape[0])
    H = repeat_center(n, nb_lines)
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            Index = Index + 1
            H[max([0, (Index - 1)*H_fact.shape[0]]):Index*H_fact.shape[0], i] = H_fact[:, 0]
            H[max([0, (Index - 1)*H_fact.shape[0]]):Index*H_fact.shape[0], j] = H_fact[:, 1]

    if center is None:
        if n<=16:
            points= [0, 0, 0, 3, 3, 6, 6, 6, 8, 9, 10, 12, 12, 13, 14, 15, 16]
            center = points[n]
        else:
            center = n
        
    H = np.c_[H.T, repeat_center(n, center).T].T
    
    return H

def repeat_center(n, repeat):
    """
    Create the center-point portion of a design matrix
    
    Parameters
    ----------
    n : int
        The number of factors in the original design
    repeat : int
        The number of center points to repeat
    
    Returns
    -------
    mat : 2d-array
        The center-point portion of a design matrix (elements all zero).
    
    Example
    -------
    ::
    
        >>> repeat_center(3, 2)
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.]])
       
    """
    return np.zeros((repeat, n))

"""
This code was originally published by the following individuals for use with
Scilab:
    Copyright (C) 2012 - 2013 - Michael Baudin
    Copyright (C) 2012 - Maria Christopoulou
    Copyright (C) 2010 - 2011 - INRIA - Michael Baudin
    Copyright (C) 2009 - Yann Collette
    Copyright (C) 2009 - CEA - Jean-Marc Martinez
    
    website: forge.scilab.org/index.php/p/scidoe/sourcetree/master/macros

Much thanks goes to these individuals. It has been converted to Python by 
Abraham Lee.
"""

import numpy as np
from pyDOE.doe_factorial import ff2n
from pyDOE.doe_repeat_center import repeat_center

__all__ = ['bbdesign']

def bbdesign(n, center=None):
    """
    Create a Box-Behnken design
    
    Parameters
    ----------
    n : int
        The number of factors in the design
    
    Optional
    --------
    center : int
        The number of center points to include (default = 1).
    
    Returns
    -------
    mat : 2d-array
        The design matrix
    
    Example
    -------
    ::
    
        >>> bbdesign(3)
        array([[-1., -1.,  0.],
               [ 1., -1.,  0.],
               [-1.,  1.,  0.],
               [ 1.,  1.,  0.],
               [-1.,  0., -1.],
               [ 1.,  0., -1.],
               [-1.,  0.,  1.],
               [ 1.,  0.,  1.],
               [ 0., -1., -1.],
               [ 0.,  1., -1.],
               [ 0., -1.,  1.],
               [ 0.,  1.,  1.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.]])
        
    """
    assert n>=3, 'Number of variables must be at least 3'
    
    # First, compute a factorial DOE with 2 parameters
    H_fact = ff2n(2)
    # Now we populate the real DOE with this DOE
    
    # We made a factorial design on each pair of dimensions
    # - So, we created a factorial design with two factors
    # - Make two loops
    Index = 0
    nb_lines = int((0.5*n*(n-1))*H_fact.shape[0])
    H = repeat_center(n, nb_lines)
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            Index = Index + 1
            H[max([0, (Index - 1)*H_fact.shape[0]]):Index*H_fact.shape[0], i] = H_fact[:, 0]
            H[max([0, (Index - 1)*H_fact.shape[0]]):Index*H_fact.shape[0], j] = H_fact[:, 1]

    if center is None:
        if n<=16:
            points= [0, 0, 0, 3, 3, 6, 6, 6, 8, 9, 10, 12, 12, 13, 14, 15, 16]
            center = points[n]
        else:
            center = n
        
    H = np.c_[H.T, repeat_center(n, center).T].T
    
    return H

"""
This code was originally published by the following individuals for use with
Scilab:
    Copyright (C) 2012 - 2013 - Michael Baudin
    Copyright (C) 2012 - Maria Christopoulou
    Copyright (C) 2010 - 2011 - INRIA - Michael Baudin
    Copyright (C) 2009 - Yann Collette
    Copyright (C) 2009 - CEA - Jean-Marc Martinez
    
    website: forge.scilab.org/index.php/p/scidoe/sourcetree/master/macros

Much thanks goes to these individuals. It has been converted to Python by 
Abraham Lee.
"""

import re
import numpy as np

__all__ = ['np', 'fullfact', 'ff2n', 'fracfact']

def fullfact(levels):
    """
    Create a general full-factorial design
    
    Parameters
    ----------
    levels : array-like
        An array of integers that indicate the number of levels of each input
        design factor.
    
    Returns
    -------
    mat : 2d-array
        The design matrix with coded levels 0 to k-1 for a k-level factor
    
    Example
    -------
    ::
    
        >>> fullfact([2, 4, 3])
        array([[ 0.,  0.,  0.],
               [ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 1.,  1.,  0.],
               [ 0.,  2.,  0.],
               [ 1.,  2.,  0.],
               [ 0.,  3.,  0.],
               [ 1.,  3.,  0.],
               [ 0.,  0.,  1.],
               [ 1.,  0.,  1.],
               [ 0.,  1.,  1.],
               [ 1.,  1.,  1.],
               [ 0.,  2.,  1.],
               [ 1.,  2.,  1.],
               [ 0.,  3.,  1.],
               [ 1.,  3.,  1.],
               [ 0.,  0.,  2.],
               [ 1.,  0.,  2.],
               [ 0.,  1.,  2.],
               [ 1.,  1.,  2.],
               [ 0.,  2.,  2.],
               [ 1.,  2.,  2.],
               [ 0.,  3.,  2.],
               [ 1.,  3.,  2.]])
               
    """
    n = len(levels)  # number of factors
    nb_lines = np.prod(levels)  # number of trial conditions
    H = np.zeros((nb_lines, n))
    
    level_repeat = 1
    range_repeat = np.prod(levels)
    for i in range(n):
        range_repeat //= levels[i]
        lvl = []
        for j in range(levels[i]):
            lvl += [j]*level_repeat
        rng = lvl*range_repeat
        level_repeat *= levels[i]
        H[:, i] = rng
     
    return H
    
################################################################################

def ff2n(n):
    """
    Create a 2-Level full-factorial design
    
    Parameters
    ----------
    n : int
        The number of factors in the design.
    
    Returns
    -------
    mat : 2d-array
        The design matrix with coded levels -1 and 1
    
    Example
    -------
    ::
    
        >>> ff2n(3)
        array([[-1., -1., -1.],
               [ 1., -1., -1.],
               [-1.,  1., -1.],
               [ 1.,  1., -1.],
               [-1., -1.,  1.],
               [ 1., -1.,  1.],
               [-1.,  1.,  1.],
               [ 1.,  1.,  1.]])
       
    """
    return 2*fullfact([2]*n) - 1

################################################################################

def fracfact(gen):
    """
    Create a 2-level fractional-factorial design with a generator string.
    
    Parameters
    ----------
    gen : str
        A string, consisting of lowercase, uppercase letters or operators "-"
        and "+", indicating the factors of the experiment
    
    Returns
    -------
    H : 2d-array
        A m-by-n matrix, the fractional factorial design. m is 2^k, where k
        is the number of letters in ``gen``, and n is the total number of
        entries in ``gen``.
    
    Notes
    -----
    In ``gen`` we define the main factors of the experiment and the factors
    whose levels are the products of the main factors. For example, if
    
        gen = "a b ab"
    
    then "a" and "b" are the main factors, while the 3rd factor is the product
    of the first two. If we input uppercase letters in ``gen``, we get the same
    result. We can also use the operators "+" and "-" in ``gen``.
    
    For example, if
    
        gen = "a b -ab"
    
    then the 3rd factor is the opposite of the product of "a" and "b".
    
    The output matrix includes the two level full factorial design, built by
    the main factors of ``gen``, and the products of the main factors. The
    columns of ``H`` follow the sequence of ``gen``.
    
    For example, if
    
        gen = "a b ab c"
    
    then columns H[:, 0], H[:, 1], and H[:, 3] include the two level full
    factorial design and H[:, 2] includes the products of the main factors.
    
    Examples
    --------
    ::
    
        >>> fracfact("a b ab")
        array([[-1., -1.,  1.],
               [ 1., -1., -1.],
               [-1.,  1., -1.],
               [ 1.,  1.,  1.]])
       
        >>> fracfact("A B AB")
        array([[-1., -1.,  1.],
               [ 1., -1., -1.],
               [-1.,  1., -1.],
               [ 1.,  1.,  1.]])
        
        >>> fracfact("a b -ab c +abc")
        array([[-1., -1., -1., -1., -1.],
               [ 1., -1.,  1., -1.,  1.],
               [-1.,  1.,  1., -1.,  1.],
               [ 1.,  1., -1., -1., -1.],
               [-1., -1., -1.,  1.,  1.],
               [ 1., -1.,  1.,  1., -1.],
               [-1.,  1.,  1.,  1., -1.],
               [ 1.,  1., -1.,  1.,  1.]])
       
    """
    # Recognize letters and combinations
    A = [item for item in re.split('\-?\s?\+?', gen) if item]  # remove empty strings
    C = [len(item) for item in A]
    
    # Indices of single letters (main factors)
    I = [i for i, item in enumerate(C) if item==1]
    
    # Indices of letter combinations (we need them to fill out H2 properly).
    J = [i for i, item in enumerate(C) if item!=1]
    
    # Check if there are "-" or "+" operators in gen
    U = [item for item in gen.split(' ') if item]  # remove empty strings
    
    # If R1 is either None or not, the result is not changed, since it is a
    # multiplication of 1.
    R1 = _grep(U, '+')
    R2 = _grep(U, '-')
    
    # Fill in design with two level factorial design
    H1 = ff2n(len(I))
    H = np.zeros((H1.shape[0], len(C)))
    H[:, I] = H1
    
    # Recognize combinations and fill in the rest of matrix H2 with the proper
    # products
    for k in J:
        # For lowercase letters
        xx = np.array([ord(c) for c in A[k]]) - 97
        
        # For uppercase letters
        if np.any(xx<0):
            xx = np.array([ord(c) for c in A[k]]) - 65
        
        H[:, k] = np.prod(H1[:, xx], axis=1)
    
    # Update design if gen includes "-" operator
    if R2:
        H[:, R2] *= -1
        
    # Return the fractional factorial design
    return H
    
def _grep(haystack, needle):
    try:
        haystack[0]
    except (TypeError, AttributeError):
        return [0] if needle in haystack else []
    else:
        locs = []
        for idx, item in enumerate(haystack):
            if needle in item:
                locs += [idx]
        return locs



import time
import pandas as pd
import numpy as np
# ========================
# Main execution function
# ========================

def main_choice():
    print("-"*60+"\n"+"Welcome to the Main Menu\n"+"-"*60+"\n")
    print("1) Design of Experiment\n2) Optimization\n")
    user_choice = int(input("Please select n option number from above: "))
    print()
    inputs = int(input("Please enter the number of inputs of csv file (digits only): "))
    return user_choice, inputs
    
def execute_main(num_inputs):
    """
    Main function to execute the program.
    Calls "user_input" function to receive the choice of the DOE user wants to build and to read the input CSV file with the ranges of the variables. Thereafter, it calls the "generate_DOE" function to generate the DOE matrix and a suitable filename corresponding to the user's DOE choice. Finally, it calls the "write_CSV" function to write the DOE matrix (a Pandas DataFrame object) into a CSV file on the disk, and prints a message indicating the filename.
    """
    doe_choice, infile = user_input()
    df, filename = generate_DOE(doe_choice,infile)
    if type(df)!=int or type(filename)!=int:
        flag=write_csv(df,filename)
        if flag!=-1:
            print("\nWorking on the DOE...",end=' ')
            time.sleep(2)
            print("DONE!!")
            time.sleep(0.5)
            print(f"Output has been written to the file: {filename}.csv\n")
            ch = input("Do you want to continue optimization? Y/N: ")
            print()
            if ch == "Y" or ch == "y":
                genetic(num_inputs)
            elif ch == "N" or ch == "n":
                dataframe = pd.read_csv(infile)
                columns = np.array(dataframe.keys())
                pareto(dataframe, columns, num_inputs)
                print("Thank you for using our Tool")  
            else:
                print("Invalid choice")
        else:
            print("\nError in writing the output. \nIf you have a file open with same filename, please close it before running the command again!")


#=====================================================
# Main UX with simple information about the software
#=====================================================
           
option, num_inputs = main_choice()
print()
if option == 1:
    execute_main(num_inputs)
elif option == 2:
    genetic(num_inputs)
else:
    print("Invalid input")
