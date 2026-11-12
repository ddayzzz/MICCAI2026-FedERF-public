export PYTHONPATH=$PYTHONPATH:${PWD}
CUDA_VISIBLE_DEVICES=5 jupyter lab --ip='*' --NotebookApp.token='' --NotebookApp.password='' --port 13982
