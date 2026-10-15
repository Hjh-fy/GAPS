from pathlib import Path
import numpy as np
from tools.preprocessor_time_bin_diagnostic import _max_run

def test_time_bin_max_missing_run():
    assert _max_run(np.array([False, True, True, False, True])) == 2

def test_time_bin_module_is_diagnostic_only():
    text = (Path(__file__).parents[1] / 'tools' / 'preprocessor_time_bin_diagnostic.py').read_text(encoding='utf-8')
    assert 'does not change the production time-aware preprocessor' in text
