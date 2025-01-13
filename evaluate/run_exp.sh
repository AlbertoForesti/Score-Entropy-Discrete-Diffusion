export HYDRA_FULL_ERROR=1
python -m run --config-name config --multirun +estimator=[infonce] +distribution=[categorical_big_alphabet]