export HYDRA_FULL_ERROR=1
python -m run --config-name config_genomics --multirun +estimator=[infosedd] +distribution=[humanvsworm] distribution.p_random=0.5