import os
import numpy as np
import json
from detectron2.structures import BoxMode
import cv2

from detectron2 import model_zoo

import sys
sys.path.append('/data2/epfl_rgbd_pedestrians/updated-ground-truth/')
from data import get_epfl_dicts

from detectron2.data import DatasetCatalog, MetadataCatalog
_datasets = ["train_relabel_eavise", "val_relabel_eavise"]
for d in _datasets:
    DatasetCatalog.register("epfl_" + d, lambda d=d: get_epfl_dicts("/data2/epfl_rgbd_pedestrians/updated-ground-truth/" + d))
    MetadataCatalog.get("epfl_" + d).set(thing_classes=["person"])
epfl_metadata = MetadataCatalog.get("epfl_"+_datasets[0])
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
#cfg.merge_from_file(model_zoo.get_config_file("Misc/mask_rcnn_R_50_FPN_1x_dconv_c3-c5.yaml"))
#cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/mask_rcnn_R_50_FPN_1x_dconv_c3-c5.yaml")
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
cfg.OUTPUT_DIR="/data2/epfl_rgbd_pedestrians/output"

cfg.DATASETS.TRAIN = ("epfl_"+_datasets[0],)
cfg.DATASETS.TEST =  ()#"epfl_"+_datasets[1],)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.MASK_ON = False
cfg.SOLVER.IMS_PER_BATCH = 2
#cfg.TEST.EVAL_PERIOD = 10
#cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
#cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)                                             
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
print(cfg.dump()) 

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


from detectron2.engine import DefaultPredictor

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("epfl_"+_datasets[1], )
predictor = DefaultPredictor(cfg)

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("epfl_"+_datasets[1], cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "epfl_"+_datasets[1])
inference_on_dataset(trainer.model, val_loader, evaluator)
