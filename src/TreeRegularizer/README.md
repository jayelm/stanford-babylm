### STEPS:
1. Install conda environment: 

```
conda env create -f environment.yml
conda activate tree-reg
pip install -e .
```

2. Run `train_transformers.py` as
```python train_transformers.py --dataset cogs --save_dir /path/to/save/dir --encoder_depth 6``` 
3. Pick a checkpoint (/path/to/save/dir/ckpt.pickle) and run
```python run_tree_projections.py --model_path /path/to/save/dir/ckpt.pickle --encoder_depth 6 --data cogs```



