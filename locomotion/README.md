


## D4RL experiments

The code base is adapted from [CEP-energy-guided-diffusion](https://github.com/thu-ml/CEP-energy-guided-diffusion#)

### Requirements
Installations of [PyTorch](https://pytorch.org/), [MuJoCo](https://github.com/deepmind/mujoco), and [D4RL](https://github.com/Farama-Foundation/D4RL) are needed.

### Model
We can download the pretrained model at [download url](https://drive.google.com/drive/folders/1snFcmcJaalcCWW9roBjeCjpWjpCeDM_P?usp=drive_link), the models will be stored at `./models_rl/`.


### Inference
To evaluate test-time search in our method, run `exp_tts.py`. To evaluate QGPO with time dependent energy guidance, run `exp_cep.py`

### Policy distillation
To run policy distillation, run `train_behavior.py`. To evaluate the finetuned model, run `eval_ft.py`