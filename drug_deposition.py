"""
This script runs a finite differences scheme for predicting 
drug deposition in an arterial wall.

It takes a mutli compartment (or single) model of the wall 
and descritizes the mass transfer constants for each tissue
along the solution mesh.

example usage:
# run the simulation using the 4 compartment model from vafai 2006
# for 6000 seconds, with a delivery time of 4 minutes
# use a time step of 0.002 seconds and a radial step of 1 um 
python drug_dieposition.py --model vafai4 --time 6000 --delivery_time 240 --dt 0.002 --dr 1

The script will output a csv file with the results as well as a
python pickle of the simulation class.
"""

# In[6]:
import sys, os
import numpy as np
import pandas as pd
import pickle
import argparse

# # Parameters
# 
# Multilayer models the the arterial wall taken from Jesionek 2014.
# 
# "Effects of shear stress on low-density lipoproteins (LDL) transport in the multi-layered arteries"
# 
# The class object allow us to easy access all variables.
# Define constants and wall models as lists ala Jesionek.


class Vafai_4_Layer(object):
    """ 
        L. Ai, K. Vafai,
        International Journal of Heat and Mass Transfer
        49 (2006)
        
        Had to keep the porosity (epsilon) and the p
        partition coefficient from Abraham et al.
        
        Dt is for LDL ... need to approximate it to the drug
        
            
    """
    
    # Global
    Df = 50
    vessel_radius = 2.00E3 
    dp = 9332.6
    
    # Variable  
    names       = [  'endothel'  , 'intima' , 'IEL'    ,'media'   ]
    L           = [   2.         , 10.      , 2.       , 200.     ] # 
    Dt          = [   8.514E-13  , 5.0E-8   , 3.18E-11 , 5.0E-10  ] # cm^2/s
    epsilon     = [   0.0005     , 0.96     , 0.004    , 0.15     ] # D'Less
    alpha       = [   1.145E-2   , 1.7084E-1, 1.7084E-1, 1.34E-1  ] # D'less 
    gamma       = [   38         , 38       , 38       , 38       ] # D'less --> from abraham 
    K           = [   3.2172e-17 , 2.2e-12  , 3.18e-15 , 2.0E-14  ] # cm^2  
    mu          = [   0.72e-3    , 0.72e-3  , 0.72e-3  , 0.72e-3  ] # g/cm^2 s
    
    # convert units
    Dt = [i * 1E8 for i in Dt]
    K  = [i * 1E8 for i in K]
    
    
    # convert to drug units
    dta = 5.0E-11 # m^2/s
    dta = dta * 1E12 # um^2/s

    # Assume the reference abraham is for the intima
    Dt_new = []
    for i in range(len(Dt)):
        Dt_new.append( Dt[i]*(dta/Dt[1]) )

    Dt = Dt_new

class Abraham_1_Layer(object):
    """ Single Phase testing parameters used in Abraham et al.
    """
    
    # Global Parameters
    
    # Flux - VR
    # Diffusivity of drug in fluid
    # vessel size
    # Pressure drop
    VR = 37.5 # um^2/s?
    Df = 50 # um^2/s
    vessel_radius = 2.00E3 # um
    dp = 4784 # Pa
    
    # Parameters of multi layer wall
    # Wall thicknes (r) - L [um]
    # Diffusivity - D [um^2/s]
    # porosity - epsilon  [D'less]
    # binding - alpha (1-reflection coefficient) [D'less]
    # partition coefficient - gamma [D'less]
        
    name        = [ 'wall' ]
    L           = [ 214    ] # this is to match the new model
    Dt          = [ 6E-13    ] # Diffusivity  m^2/s
    epsilon     = [ 0.61   ] # porosity [D'less]
    alpha       = [ 0.01   ] # binding coefficeint [D'less]
    gamma       = [ 38     ] # parition coefficient 
    K           = [ 2E-18 ] # permeability m^2/s
    mu          = [ 1.4E-3 ] # dynamic viscosity Pa-s


    Dt = [i * 1E12 for i in Dt]
    K  = [i * 1E12 for i in K]

class Drug_Diffusion(object):
    # # Simulation Class
    # The simulation class is the work horse of the implementation.  It does a number of things:
    # 1) takes in an arterial wall parameter model defined above
    # 2) Descritizes the wall in radius and time, as well as parameter space across the radius
    # 3) Solves the drug diffusion model

    def __init__(self, parameters, options=None):
        # initialize object with parameters and options
        self.options = options
        self.p = parameters
        self.t = None
        self.f = None
        
    def descritize(self, N = 1000, M=1000, total_time=1000, delivery_time = 1000, dt=None, dr=None):
        """ 
        Create spacital grids for radial direction, time, and constants based on the wall model.
        """
        
        # get the total wall length 
        self.l = np.sum(self.p.L) 
    
        # override provided N if dr is provided ...
        # TODO: this is odd behavior fix it after testing
        if dr:
            self.dr = dr
            self.N = int(self.l / self.dr)
        else:
            self.N = N
            self.dr = float(self.l) / self.N
        
        # get spacial grid
        self.r = np.linspace(self.p.vessel_radius, self.p.vessel_radius + self.l, self.N)
        
        # TODO: mind your units
        # convert to meters
        self.r = self.r 
 
        # descritize time
        self.total_time = total_time
        if dt:
            self.dt = dt
            self.M = int(total_time / dt)
        else:
            self.M = M
            self.dt = float(self.total_time) / self.M

        # store delivery time
        self.delivery_time = delivery_time
        
        # init vectors for constants that change along radial direction
        self.Dt = np.ones(self.N)
        self.epsilon = np.ones(self.N)
        self.alpha = np.ones(self.N)
        self.gamma = np.ones(self.N)
        self.mu = np.ones(self.N)
        self.K = np.ones(self.N)

        # normalize the length of each layer so the sum is over the spactial grid
        # then get thew spacial grid boundary points
        self.layers=[0]+list( np.ceil( ( self.N * ( np.cumsum(self.p.L) / self.l ) ) ).astype(np.int32) )
        
        # make vectors of constants according the parameter wall model supplied
        for i,(a,b) in enumerate(zip(self.layers[:],self.layers[1:])): 
            self.Dt[a:b] = self.p.Dt[i] 
            self.epsilon[a:b] = self.p.epsilon[i] 
            self.alpha[a:b] = self.p.alpha[i]
            self.gamma[a:b] = self.p.gamma[i]
            self.mu[a:b] = self.p.mu[i]
            self.K[a:b] = self.p.K[i]
        
    def solve(self, fluid_bc = [1,0], tissue_bc=[1,0]):
        """
        Function to solve the diffusion equation
        """
        # create tissue mesh
        t = np.zeros((self.M,self.N)) # rows are time, columns are radius
        
        # create fluid mesh
        f = np.zeros((self.M,self.N))
        
        # set the boundary condition of delivery for some time as a constant  stream of drug in the center of the fluid
        delivery_index = int(np.ceil((float(self.delivery_time) / self.total_time)* self.M))
        f[:delivery_index,0] = fluid_bc[0]
        t[:delivery_index,0] = fluid_bc[0]*self.p.gamma[0]

        out_handle = open(os.path.join(options.output, "progress.txt"), "a")

        # Loop through time and space, solving the finite differences model
        # by population the solution matricies f and t
        for i in range(1,self.M-1): # time
            if i % (100/self.dt) == 0:
                out.write("Complete {} time steps.\n".format(int(i*self.dt)))
            for j in range(1,self.N-1): # space

                # get wall model parameters for current spacial location
                r = self.r[j]
                alpha = self.alpha[j]
                gamma = self.gamma[j]
                epsilon = self.epsilon[j]
                Dt = self.Dt[j]
                mu = self.mu[j]
                K = self.K[j]
                
                # calculate velocity
                u = K * self.p.dp / mu / np.log(( self.p.vessel_radius + self.l     ) / self.p.vessel_radius )
                
                # update fluid using equation from the paper
                f[i, j] = f[i-1,j] +                                         u / (epsilon * self.r[j]) * (self.dt /  self.dr) * (f[i-1,j-1] - f[i-1,j])  +                                         self.p.Df * self.dt / (2 * self.r[j] * self.dr**2) * (                                             ( self.r[j+1] + self.r[j] ) * (f[i-1,j+1] - f[i-1,j]) -                                             ( self.r[j] + self.r[j-1] ) * (f[i-1,j] - f[i-1,j-1])) +                                         self.dt * alpha * (t[i-1,j] -  gamma * f[i-1,j] )
                            
                # update tissue
                t[i, j] = t[i-1, j] +                                         (Dt * self.dt)/( 2 * self.r[j] * self.dr**2 ) * (                                             ( self.r[j+1] + self.r[j] ) * (t[i-1,j+1] - t[i-1,j]) -                                             ( self.r[j] + self.r[j-1] ) * (t[i-1,j] - t[i-1,j-1])) +                                         self.dt * alpha * ( gamma * f[i-1,j] - t[i-1,j])
                

        out_handle.close()
        # store results
        self.t = t
        self.f = f
       
def my_melt(sim, fluid=True, sub_sample = True):
    """
    # function to melt the solution matrix from a simulation
    """
    if fluid: # if the fluid flag is set then prep the solution for the fluid matrix
        df = pd.DataFrame(sim.f)
    else: # otherwise do the tissue
        df = pd.DataFrame(sim.t) 
    
    # if we run the model with a small time step then optionally subsample the time (rows)
    # so the plotting doesn't break/take forever
    if sub_sample and (sim.dt < 1):
        df = df.iloc[::int(1/sim.dt),:]
        df.index = list(np.arange(0,sim.total_time)) # add time labels to rows
    else:
        df.index = list(np.arange(0,sim.total_time,sim.dt)) # add time labels to rows

    # add spacial column labels
    df.columns = list(np.arange(0, sim.l, sim.dr))
    # melt matrix 
    df = df.stack().reset_index() 
    # add column labels
    df.columns = ["time", "radius", "concentration"]
            
    return df

def merge_data(simd, sim2):
    """
    Functin to merge two runs of the drug diffusion simulation together
    """
    # get melted versions of solution matrix 
    default_f = my_melt(simd, fluid=True)
    default_t = my_melt(simd, fluid=False)
    multi_f = my_melt(sim2, fluid=True)
    multi_t = my_melt(sim2, fluid=False)

    # add wall model identifiers to the dataframes
    default_f['model'] = "default"
    default_t['model'] = "default"
    multi_f['model'] = "multi"
    multi_t['model'] = "multi"

    # add fluid/tissue identifier
    default_f['medium'] = "fluid"
    default_t['medium'] = "tissue"
    multi_f['medium'] = "fluid"
    multi_t['medium'] = "tissue"

    # make a list of the resulting data frames
    frames = [default_f, default_t, multi_f, multi_t]

    # combine all the results into one dataframe
    data = pd.concat(frames)

    return data

def main(options):
    # # Run Simulations
    # write options to file for safe storage
    out_handle = open(os.path.join(options.output, "progress.txt"), "w")
    out_handle.write("Staring with paramters: {}\n".format(options))
    out_handle.close()

    # figure out which model we want to use
    if options.model == "default":
        parameters = Abraham_1_Layer()
    if options.model == "vafai4":
        parameters = Vafai_4_Layer()

    # initialize the drug diffusion simulation class with the wall model
    sim = Drug_Diffusion(parameters, options=options)
    # run the descritization, initializing the parameters, space, and time, matrices
    sim.descritize( total_time=options.time,
                    delivery_time=options.delivery_time,
                    dt=options.dt,
                    dr=options.dr)
    # run the finite differences scheme
    sim.solve()

    # get melted the two tissue types together
    multi_f = my_melt(sim, fluid=True)
    multi_t = my_melt(sim, fluid=False)

    # add wall model identifiers to the dataframes
    multi_f['model'] = options.model
    multi_t['model'] = options.model

    # add fluid/tissue identifier
    multi_f['medium'] = "fluid"
    multi_t['medium'] = "tissue"

    # make a list of the resulting data frames
    frames = [multi_f, multi_t]

    # combine all the results into one dataframe
    data = pd.concat(frames)

    # get some parameters into the name space so we can pass then to R
    data.to_csv(os.path.join(options.output,'sim.{}.csv'.format(options.model)), sep=",",header=True)
    
    # Store data
    with open(os.path.join(options.output,'sim.{}.p'.format(options.model)), 'wb') as output:
        pickle.dump(sim, output)   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage = globals()['__doc__'])

    # flag based reporting options
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Verbose output")      
    parser.add_argument("-o", "--output", action="store", default="./", help="output directory")
    parser.add_argument("--model",  action="store", default='default', help="Specify the wall model to use")
    parser.add_argument("--delivery_time", action="store", default=120, type=float, help="Amount of time to deliver the drug")
    parser.add_argument("--dt", action="store", default=0.002, type=float, help="time step")
    parser.add_argument("--dr", action="store", default=1, type=float, help="radial step")
    parser.add_argument("--time", action="store", default=100, type=int, help="Total Run Time")

    # parse arguments
    options = parser.parse_args()   

    exit_code = main(options)

    sys.exit(exit_code)


