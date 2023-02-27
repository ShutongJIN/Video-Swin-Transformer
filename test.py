import torch
from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

model = init_recognizer(config_file, device=device)
# inference the demo video
inference_recognizer(model, 'test_demo/demo.mp4', 'test_demo/label_map_k400.txt')
