from scipy.interpolate import interp1d

import numpy as np
import pandas as pd
import unittest

import tools



class Test_tools(unittest.TestCase):
    
    '''
    

    '''
    
    
    def derivative_aux(self, x, y, g_analytical, precision, numPointsForTest=50, error='rel', num_multiply=10):
        '''
        Tests if the relative error in the derivative is smaller than a given precision.
        
        TODO: This test seems very confused. Maybe itself is prone to errors.
              It is a good idea to simplify it in a future time.
        '''
        
        if numPointsForTest < 10:
            raise Exception("Parameter numPointsForTest should be bigger than 10.")
        
        if not error in ['rel', 'abs']:
            raise Exception("Parameter error should be in ['rel', 'abs'].")
        
        # Calculate the numerical derivative interpolator.
        g = tools.derivative(x, y, num_multiply=num_multiply)
        
        # Test
        out = numPointsForTest//10
        X = np.linspace(min(x),max(x), numPointsForTest)[out:-out]

        Y_analytical = np.array([g_analytical(i) for i in X])
        Y = np.array([g(i) for i in X])
        
        if error == 'rel':
            rel_err = np.abs((Y_analytical-Y)/Y_analytical)
        else:
            rel_err = np.abs((Y_analytical-Y))
        
        self.assertEqual(sum(rel_err < precision), len(X))
        
        
        
        

    def test__derivative(self):

        # Functions that cannot be locally well approximated by cubic functions have 
        # poor derivative calculated by this method (again for input with few points).

        # Cosine function with just a few points is hard.
        # NOTE: The relative error is immense for cosine function close to x equals to pi multipliers. 
        #       Therefore, I use the absolute error.
        x = np.linspace(0, 11, num=12)
        y = np.cos(x)
        g_analytical = lambda x: -np.sin(x)
        
        self.derivative_aux(x, y, g_analytical, 5e-2, 100, error='abs')
        
        # More input points, smaller error.
        x = np.linspace(0, 11, num=102)
        y = np.cos(x)
        g_analytical = lambda x: -np.sin(x)
        
        self.derivative_aux(x, y, g_analytical, 1.1e-5, error='abs')
        
        # As the interpolations are cubic, let us test a fourth-degree polynomial function.
        x = np.linspace(0, 2, num=51)
        y = -x**4 + 3*x**3 - 3*x**2 + 2
        g_analytical = lambda x: -4*x**3 + 9*x**2 - 6*x
        self.derivative_aux(x, y, g_analytical, 1.25e-5, error='rel')
        
        x = np.linspace(-5, 5, num=101)
        y = -x**4 + 3*x**3 - 3*x**2 + 2
        g_analytical = lambda x: -4*x**3 + 9*x**2 - 6*x
        self.derivative_aux(x, y, g_analytical, 1.1e-4, 100, error='rel')


            
    def test__express_measure_with_error(self):
        
        # I tested the function more extensively while I was developing it. Now I will leave it for another time. 
         
        self.assertEqual(tools.express_measure_with_error(3012,99.3), (3000.0, 100.0))
        self.assertEqual(tools.express_measure_with_error(3012,91.3), (3010.0, 90.0))
        self.assertEqual(tools.express_measure_with_error(0.666,0.000088, 'A'), 'A = 0.66600 ± 0.00009')

    
    
    # TODO: Include tests for intersection(). The functions regression() and regression_power_law() are so simple that they do not need tests.

        
if __name__ == '__main__':
    full = pd.read_csv('test__full.csv', index_col=0)
    unittest.main()        
        
        
        
        
        
