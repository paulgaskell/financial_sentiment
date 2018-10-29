
from financial_sentiment_analysis import MatrixOperations, LMNDataReader 
import numpy as np
import datetime 



def test_LMNDataReader():
    dr = LMNDataReader('data_for_financial_sentiment_paper.zip')
    for i in dr.nt_data():
        assert type(i[0]) == datetime.date and type(i[1]) == str and type(i[2]) == float 
        
test_LMNDataReader()

def test_diff():
    """
    if you diff the same numbers you get 0
    """
    # matrix [[1,2],[1,2]....]
    MATRIX = np.ones((10, 2))
    MATRIX.T[1] += 1

    mo = MatrixOperations()
    x = mo.diff(MATRIX, 0)
    x = mo.diff(MATRIX, 1)
    assert sum(sum(x)) == 0
    
def test_var():
    """
    test the dims are correct and start and end are as expected 
    """
    # matrix [[1],[2]....]
    MATRIX = np.ones((10, 1))
    MATRIX.T[0] = np.arange(10)    
    test_arr = np.array([9., 8., 7., 6., 5.])
    
    mo = MatrixOperations()
    x = mo.var(MATRIX, 0, [1, 2, 3, 4])
    print(x)
    assert x.shape == (10, 5) and sum(x[0]) == 0 and sum(x[-1]-test_arr) == 0

def test_append():
    """
    if you diff the same numbers you get 0
    """
    # matrix [[1,2],[1,2]....]
    MATRIX = np.ones((10, 2))
    MATRIX.T[1] += 1

    mo = MatrixOperations()
    x = mo.append(MATRIX, MATRIX)
    assert x.shape == (20, 2) 
    