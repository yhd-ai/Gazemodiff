#GazeMoDiff

Code for "GazeMoDiff: Gaze-guided Diffusion Model for
Stochastic Human Motion Prediction"





### Environment Setup

```
sh install.sh
```

### DATA PROCESSING

1. Download the original mogaze dataset from https://humans-to-robots-motion.github.io/mogaze/ and Download the original gimo dataset from https://github.com/y-zheng18/GIMO.

2. For MoGaze:

```
python motion_generation/mogaze_code/mogaze_preprocessing.py
```

For GIMO:

```
python motion_generation/gimo_code/gimo_preprocessing.py
```

3. Obtain the multimodal indices.

For MoGaze:

```
python humanmac/get_multimodal.py --dataset mogaze_withcontext
```

For GIMO:

python humanmac/get_multimodal.py --dataset gimo_withcontext

The processed data should be placed in "./data"

3.  

## Training
For MoGaze:

```
python main.py --cfg mogaze_withcontext --mode train
```

For GIMO:

```
python main.py --cfg gimo_withcontext --mode train
```

## Visualization of Motion Prediction

For MoGaze:

```
python main.py --cfg mogaze_withcontext  --mode pred --vis_row 1 --vis_col 10 --ckpt MODEL_PATH --ckpt_gcn GCNMODEL_PATH
```

For GIMO:

```
python main.py --cfg gimo_withcontext  --mode pred --vis_row 1 --vis_col 10 --ckpt MODEL_PATH  --ckpt_gcn GCNMODEL_PATH

## Evaluation

Evaluate on MoGaze:

```
python main.py --cfg mogaze_withcontext --mode eval --ckpt MODEL_PATH  --ckpt_gcn GCNMODEL_PATH
```

Evaluate on GIMO:

```
python main.py --cfg gimo_withcontext --mode eval --ckpt MODEL_PATH --ckpt_gcn GCNMODEL_PATH
```

Contact at: k610215095@stu.xjtu.edu.cn