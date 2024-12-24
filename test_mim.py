from mmdet.apis import init_detector, inference_detector

config_file = 'faster_rcnn_r50_fpn_1x_coco.py'

checkpoint_file = 'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

inference_detector(model, 'demo/demo.jpg')
