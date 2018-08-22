from scipy.interpolate import interp1d

import numpy as np
import pandas as pd
import unittest

import tools



class Test_tools(unittest.TestCase):
    
    '''
    

    '''
    
    
    def derivative_aux(self, x, y, dy_dx_analytical, precision, num_multiply=4):
        '''
        Tests if the relative error in the derivative is smaller than a given precision.
        '''
        
        X, dy_dx = tools.derivative(x, y, num_multiply=num_multiply)
        
        analytical = pd.Series(dy_dx_analytical, x)
        numerical = pd.Series(dy_dx, X)
        
        for xi in analytical.index:
            if xi in numerical.index and xi != 0:
                self.assertLess(np.abs((analytical[xi] - numerical[xi])/analytical[xi]), precision)
        
        
        
        

    def test__derivative(self):

        # Functions that cannot be locally well approximated by cubic functions have 
        # poor derivative calculated by this method (again for input with few points).

        # Cosine function with just a few points is hard.
        # Precision only at the first decimal place.
        x = np.linspace(0, 11, num=12)
        y = np.cos(x)
        dy_dx_analytical = -np.sin(x)
        
        self.derivative_aux(x, y, dy_dx_analytical, 1e-1)
        
        # More points, smaller relative error.
        x = np.linspace(0, 11, num=102)
        y = np.cos(x)
        dy_dx_analytical = -np.sin(x)
        
        self.derivative_aux(x, y, dy_dx_analytical, 1e-3)
        
        
        # More given points and more points in the interpolation, smaller relative error.       
        self.derivative_aux(x, y, dy_dx_analytical, 1e-5, num_multiply=10)
        
        
        x = np.linspace(-11, 11, num=11)
        y = -x**4 + 3*x**3 - 3*x**2 + 2
        dy_dx_analytical = -4*x**3 + 9*x**2 - 6*x
        self.derivative_aux(x, y, dy_dx_analytical, 1e-2, num_multiply=10)
        
        x = np.linspace(-11, 11, num=101)
        y = -x**4 + 3*x**3 - 3*x**2 + 2
        dy_dx_analytical = -4*x**3 + 9*x**2 - 6*x
        self.derivative_aux(x, y, dy_dx_analytical, 1e-5, num_multiply=10)

            
        




        
if __name__ == '__main__':
    full = pd.read_csv('test__full.csv', index_col=0)
    unittest.main()        
        
        
        
        
        
