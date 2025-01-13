export HYDRA_FULL_ERROR=1
python -m run --config-name config --multirun +estimator=[infosedd] +distribution=[categorical_big_alphabet]