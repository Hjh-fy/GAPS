"""Protect ordered execution and fail-closed endpoint acceptance."""
import json
import pytest


def test_fresh_controller_can_build_source_commands_before_patch():
    import subprocess, sys
    command="from scripts.run_fedongas27 import install_fl_builder; from gaps_flower.fedongas27_protocol import run_matrix; from scripts import run_iotj_final_classification_le1 as f; install_fl_builder(run_matrix()); print(f.build_flower_commands('FONG-SRC-S42')['protocol']['method'])"
    result=subprocess.run([sys.executable,'-c',command],capture_output=True,text=True)
    assert result.returncode==0, result.stderr
    assert result.stdout.strip()=='SRC'


def test_queue_skips_verified_endpoints_but_rejects_partial_runs(tmp_path):
    from scripts.run_fedongas27 import pending_runs
    rows=[{'experiment_id':'A'},{'experiment_id':'B'}]
    a=tmp_path/'runs/A'; a.mkdir(parents=True)
    (a/'FINAL_ENDPOINT.json').write_text('{}')
    assert pending_runs(tmp_path,rows,lambda row: None)==[rows[1]]
    (tmp_path/'runs/B').mkdir()
    with pytest.raises(RuntimeError,match='partial'):
        pending_runs(tmp_path,rows,lambda row: None)


def test_target_diagnostics_require_2500_updates_and_adam_reset():
    from scripts.run_fedongas27 import validate_target_updates
    rows=[{'block':b,'step':s,'update':(b-1)*100+s,'source_ce_enabled':'False',
           'loss_name':'target_ce','optimizer_state_before_first_update':'0' if s==1 else '',
           'encoder_grad_norm':'1.0','classifier_grad_norm':'1.0','raw_loss':'0.3'}
          for b in range(1,26) for s in range(1,101)]
    validate_target_updates(rows)
    with pytest.raises(RuntimeError): validate_target_updates(rows[:-1])
    rows[100]['optimizer_state_before_first_update']='4'
    with pytest.raises(RuntimeError,match='Adam'): validate_target_updates(rows)


def test_class_da_audit_rejects_stage_or_target_ce_activity():
    from scripts.run_fedongas27 import validate_da_audit
    value={'prototype_scope':'CLASS','prototype_count':4,'true_phase_used':False,
           'seed':43,'steps':100,'changed_tensors':2,'loss_activity':[
               {'loss_name':'source_ce','configured_weight':1,'active_steps':100},
               {'loss_name':'target_ce','configured_weight':0,'active_steps':0},
               {'loss_name':'stage_mmd','configured_weight':0,'active_steps':0}]}
    validate_da_audit(value,'CLASS',43)
    value['loss_activity'][2]['active_steps']=1
    with pytest.raises(RuntimeError,match='stage'): validate_da_audit(value,'CLASS',43)


def test_wrong_seed_does_not_pass_da_audit():
    from scripts.run_fedongas27 import validate_da_audit
    with pytest.raises(RuntimeError):
        validate_da_audit({'seed':42},'FULL',44)
