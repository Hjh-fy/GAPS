import numpy as np
from tools.preprocessor_canonical_candidate import max_run

def test_canonical_max_run():
    assert max_run(np.array([False,True,True,False,True]))==2
