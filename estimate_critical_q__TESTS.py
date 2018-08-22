import pandas as pd
import unittest

import estimate_critical_q as est

class Test_estimate_critical_q(unittest.TestCase):
    
    '''
    The test cases included here runs only for the data in the 'test__full.csv' dataset. 
    This dataset is a copy of 'bubble_filtering__8_neighbors_squared_network__MXU.csv' dataset. 
    It is not easy to include interesting test cases because it relies on visual inspection of the binder cumulant plot.
    In other words, I check that for current data the estimation is working and do my tests against this belief. 
    If in future these tests fail, I need to check the estimating method.
    
    Time for running these tests: ~ 16s.
    
    '''
    
    def prepare_data(self, v):
        
        # Select the data corresponding to the given visibility v, and set the index to N and q.
        phys_quant = full[full.v == v]
        phys_quant.set_index(['N','q'], inplace=True)
        
        return phys_quant
        
        
    # test for v = 1.00
    def test__find_qc_candidate__v_100(self):
        f = est.find_qc_candidate
        phys_quant = self.prepare_data(1.00)
        self.assertEqual(f(  400, 1600,phys_quant), 0.1425)
        self.assertEqual(f(  400, 3600,phys_quant), 0.1450)
        self.assertEqual(f(  400, 6400,phys_quant), 0.1425)
        self.assertEqual(f(  400,10000,phys_quant), 0.1425)
        self.assertEqual(f( 1600, 3600,phys_quant), 0.1450)
        self.assertEqual(f( 1600, 6400,phys_quant), 0.1450)
        self.assertEqual(f( 1600,10000,phys_quant), 0.1425)
        self.assertEqual(f( 3600, 6400,phys_quant), 0.1425)
        self.assertEqual(f( 3600,10000,phys_quant), 0.1425)
        self.assertEqual(f( 6400,10000,phys_quant), 0.1425)
        
    # test for v = 0.50
    def test__find_qc_candidate__v_050(self):
        f = est.find_qc_candidate
        phys_quant = self.prepare_data(0.50)
        self.assertEqual(f(  400, 1600,phys_quant), 0.0800)
        self.assertEqual(f(  400, 3600,phys_quant), 0.0775)
        self.assertEqual(f(  400, 6400,phys_quant), 0.0800)
        self.assertEqual(f(  400,10000,phys_quant), 0.0775)
        self.assertEqual(f( 1600, 3600,phys_quant), 0.0775)
        self.assertEqual(f( 1600, 6400,phys_quant), 0.0800)
        self.assertEqual(f( 1600,10000,phys_quant), 0.0775)
        self.assertEqual(f( 3600, 6400,phys_quant), 0.0800)
        self.assertEqual(f( 3600,10000,phys_quant), 0.0775)
        self.assertEqual(f( 6400,10000,phys_quant), 0.0725)
        

    # test for v = 0.25
    def test__find_qc_candidate__v_025(self):
        f = est.find_qc_candidate
        phys_quant = self.prepare_data(0.25)
        self.assertEqual(f(  400, 1600,phys_quant), None)
        self.assertEqual(f(  400, 3600,phys_quant), None)
        self.assertEqual(f(  400, 6400,phys_quant), None)
        self.assertEqual(f(  400,10000,phys_quant), None)
        self.assertEqual(f( 1600, 3600,phys_quant), 0.002)
        self.assertEqual(f( 1600, 6400,phys_quant), 0.002)
        self.assertEqual(f( 1600,10000,phys_quant), 0.0005)
        self.assertEqual(f( 3600, 6400,phys_quant), 0.0005)
        self.assertEqual(f( 3600,10000,phys_quant), 0.0005)
        self.assertEqual(f( 6400,10000,phys_quant), 0.0015)
        
        
        
    def test__estimate(self):
        f = est.estimate
        
        N = full.N.unique()
        
        phys_quant = self.prepare_data(0.25)
        self.assertRaises(Exception, f, N, phys_quant)
        
        phys_quant = self.prepare_data(0.30)
        qc, qc_error, _1, _2 = f(N, phys_quant)
        self.assertEqual(qc, 0.00302256579301416872)
        self.assertEqual(qc_error, 0.00022032557150805711)

        phys_quant = self.prepare_data(0.35)
        qc, qc_error, _1, _2 = f(N, phys_quant)
        self.assertEqual(qc, 0.02905032171437636063)
        self.assertEqual(qc_error, 0.00053568427926355621)

        phys_quant = self.prepare_data(0.40)
        qc, qc_error, _1, _2 = f(N, phys_quant)
        self.assertEqual(qc, 0.04839484754741192607)
        self.assertEqual(qc_error, 0.00104508006484383591)

        phys_quant = self.prepare_data(0.45)
        qc, qc_error, _1, _2 = f(N, phys_quant)
        self.assertEqual(qc, 0.06597590069717755579)
        self.assertEqual(qc_error, 0.00290916539621632062)

        phys_quant = self.prepare_data(0.50)
        qc, qc_error, _1, _2 = f(N, phys_quant)
        self.assertEqual(qc, 0.07710638368738398363)
        self.assertEqual(qc_error, 0.00086185781349962221)

        phys_quant = self.prepare_data(0.55)
        qc, qc_error, _1, _2 = f(N, phys_quant)
        self.assertEqual(qc, 0.08755080606130385967)
        self.assertEqual(qc_error, 0.00144282948641173215)

        phys_quant = self.prepare_data(0.60)
        qc, qc_error, _1, _2 = f(N, phys_quant)
        self.assertEqual(qc, 0.09886689213747222593)
        self.assertEqual(qc_error, 0.00050813063145214933)

        phys_quant = self.prepare_data(0.65)
        qc, qc_error, _1, _2 = f(N, phys_quant)
        self.assertEqual(qc, 0.10696852558941080669)
        self.assertEqual(qc_error, 0.00181690591413661466)

        phys_quant = self.prepare_data(0.70)
        qc, qc_error, _1, _2 = f(N, phys_quant)
        self.assertEqual(qc, 0.11388586081378418435)
        self.assertEqual(qc_error, 0.00069479190977075151)

        phys_quant = self.prepare_data(0.75)
        qc, qc_error, _1, _2 = f(N, phys_quant)
        self.assertEqual(qc, 0.12143688732606719438)
        self.assertEqual(qc_error, 0.00106996247680876699)

        phys_quant = self.prepare_data(0.80)
        qc, qc_error, _1, _2 = f(N, phys_quant)
        self.assertEqual(qc, 0.12769546645327287115)
        self.assertEqual(qc_error, 0.00122995149383457172)

        phys_quant = self.prepare_data(0.85)
        qc, qc_error, _1, _2 = f(N, phys_quant)
        self.assertEqual(qc, 0.13367996100319284869)
        self.assertEqual(qc_error, 0.00091320492628367534)

        phys_quant = self.prepare_data(0.90)
        qc, qc_error, _1, _2 = f(N, phys_quant)
        self.assertEqual(qc, 0.13815651808868281702)
        self.assertEqual(qc_error, 0.00089430880790931531)

        phys_quant = self.prepare_data(0.95)
        qc, qc_error, _1, _2 = f(N, phys_quant)
        self.assertEqual(qc, 0.14132816415201046589)
        self.assertEqual(qc_error, 0.00175638796857373477)

        phys_quant = self.prepare_data(1.00)
        qc, qc_error, _1, _2 = f(N, phys_quant)
        self.assertEqual(qc, 0.14198322015083814085)
        self.assertEqual(qc_error, 0.00069442673384173938)
        
        
        
if __name__ == '__main__':
    full = pd.read_csv('test__full.csv', index_col=0)
    unittest.main()
        
