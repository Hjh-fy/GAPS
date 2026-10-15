import json
import pytest
import numpy as np


def test_test_payload_cannot_open_before_model_lock(tmp_path):
    from scripts.run_fedongas_regression_cross import release_test
    with pytest.raises(RuntimeError, match='locked'):
        release_test(tmp_path)
    assert not (tmp_path/'TEST_RELEASE.json').exists()


def test_protocol_pins_matrix_and_no_new_training():
    from scripts.run_fedongas_regression_cross import protocol_spec
    p=protocol_spec()
    assert p['neural_training'] is False
    assert p['pipelines']==36 and p['shared_banks']==4 and p['final_gas_fits']==16
    assert p['alpha_candidates']==[0.,.01,.1,1.,10.,100.,1000.]
    assert p['source_prior_index_sha256']=='04e62b0a2b363fcb79cbeb60ad6c2c6c9a3dedef380b2b0f7c2aa808732d4d82'
    assert len(p['source_prior_index_sha256'])==64
    assert p['prediction_cache_sha256']=='150694bdf5df22c164d939a754460b68a2ea3ee7c5ad1d9be186332c1ba7c5de'


def test_budget_payload_uses_original_quantitative_arrays():
    from scripts.run_fedongas_regression_cross import load_calibration
    small=load_calibration(5); large=load_calibration(20)
    assert small['x'].shape==(80,50,8) and large['x'].shape==(320,50,8)
    assert np.all(small['truth']>0)
    a={m['physical_identity'] for m in small['metadata']}
    b={m['physical_identity'] for m in large['metadata']}
    assert a < b


def test_release_requires_model_and_input_lock_hashes(tmp_path):
    from scripts.run_fedongas_regression_cross import release_test, write_json
    write_json(tmp_path/'target_model_lock.json',{'banks':{}})
    write_json(tmp_path/'input_lock.json',{})
    # Partial or altered bank sets must never authorize evaluation.
    with pytest.raises(RuntimeError,match='locked'):
        release_test(tmp_path)


def test_csv_writer_preserves_roundtrip_precision_and_lf(tmp_path):
    from scripts.run_fedongas_regression_cross import write_csv, read_csv
    p=tmp_path/'roundtrip.csv'; value=1.2345678901234567
    write_csv(p,[{'x':value}])
    assert float(read_csv(p)[0]['x'])==value
    assert b'\r' not in p.read_bytes()
