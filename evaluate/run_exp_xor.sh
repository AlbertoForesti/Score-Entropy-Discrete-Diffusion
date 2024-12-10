export HYDRA_FULL_ERROR=1
python -m run --config-name config_xor --multirun +estimator=infosedd +distribution=[xor]