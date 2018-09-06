import pandas as pd
import unittest

import estimate_exponents as est

class Test_estimate_exponents(unittest.TestCase):
    
    '''
    The test cases included here runs only for the data in the 'test__full.csv' dataset. 
    This dataset is a copy of 'bubble_filtering__8_neighbors_squared_network__MXU.csv' dataset.
    I will not include extensive tests for 'estimate_exponent' method and no test at all for data collapse method 
    (this method should show "coherence" when used. In other words, the produced plots should look right).
    In this class I check that for current data the estimation is working and do my tests against this belief. 
    If in future these tests fail, I need to check the methods.
    '''
    


    def prepare_data(self, v):
        
        # Select the data corresponding to the given visibility v, and set the index to N and q.
        phys_quant = full[full.v == v]
        
        N = phys_quant.N.unique()
        
        phys_quant.set_index(['N','q'], inplace=True)
        
        return N, phys_quant
    
    
    
    # tests for v = 1.00
    
    def test__estimate__v_100(self):

        N, phys_quant = self.prepare_data(1.00)
        
        exp, exp_err, y_til_0, y_til_0_err, reg_rsquared = est.estimate_exponent('beta_nu', 0.14198322015083814, N, phys_quant, scale_var='N')
        
        self.assertEqual(exp           , 0.06577135890045809  )
        self.assertEqual(exp_err       , 0.00904406886844031  )
        self.assertEqual(y_til_0       , 0.8457961130603183   )
        self.assertEqual(y_til_0_err   , 0.061101978161997084 )
        self.assertEqual(reg_rsquared  , 0.9463198996398198   )
        
        exp, exp_err, y_til_0, y_til_0_err, reg_rsquared = est.estimate_exponent('gamma_nu', 0.14198322015083814, N, phys_quant, scale_var='N')
        
        self.assertEqual(exp           , 0.8906263066183149  )
        self.assertEqual(exp_err       , 0.05260514357833005 )
        self.assertEqual(y_til_0       , 0.04483867924746968 )
        self.assertEqual(y_til_0_err   , 0.01884112135751279 )
        self.assertEqual(reg_rsquared  , 0.989642263189781   )

        exp, exp_err, y_til_0, y_til_0_err, reg_rsquared = est.estimate_exponent('inv_nu', 0.14198322015083814, N, phys_quant, scale_var='N')
        
        self.assertEqual(exp           , 0.5729151273686817  )
        self.assertEqual(exp_err       , 0.14429340608736374 )
        self.assertEqual(y_til_0       , 0.14364864190364954 )
        self.assertEqual(y_til_0_err   , 0.16556698280213575 )
        self.assertEqual(reg_rsquared  , 0.8401259288029657  )     
        
        
        D, D_err = est.estimate_D(0.06577135890045809, 0.00904406886844031, 0.8906263066183149, 0.05260514357833005)
        
        self.assertEqual(D      , 1.022169024419231    )
        self.assertEqual(D_err  , 0.07069328131521067  )


        
if __name__ == '__main__':
    full = pd.read_csv('test__full.csv', index_col=0)
    unittest.main()        
        
        
        
        
        
