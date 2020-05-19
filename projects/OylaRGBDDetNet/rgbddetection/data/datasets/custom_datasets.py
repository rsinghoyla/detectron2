# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import io
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

import json
import numpy as np
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode




@dataclass
class DatasetInfo:
    name: str
    images_root: str
    annotations_fpath: str


DATASETS = [
    DatasetInfo(
        name="balloon_train",
        images_root="balloon/train",
        annotations_fpath="balloon/train/"
    ),
    DatasetInfo(
        name="balloon_val",
        images_root="balloon/val",
        annotations_fpath="balloon/val/"
    ),
    DatasetInfo(
        name="epfl_train_relabel_eavise",
        images_root="epfl_rgbd_pedestrians/",
        annotations_fpath="epfl_rgbd_pedestrians/updated-ground-truth/train_relabel_eavise.csv"
    ),
    DatasetInfo(
        name="epfl_val_relabel_eavise",
        images_root="epfl_rgbd_pedestrians/",
        annotations_fpath="epfl_rgbd_pedestrians/updated-ground-truth/val_relabel_eavise.csv"
    ),

]

BASE_DATASETS = []
# BASE_DATASETS = [
#     CocoDatasetInfo(
#         name="base_coco_2017_train",
#         images_root="coco/train2017",
#         annotations_fpath="coco/annotations/instances_train2017.json",
#     ),
#     CocoDatasetInfo(
#         name="base_coco_2017_val",
#         images_root="coco/val2017",
#         annotations_fpath="coco/annotations/instances_val2017.json",
#     ),
#     CocoDatasetInfo(
#         name="base_coco_2017_val_100",
#         images_root="coco/val2017",
#         annotations_fpath="coco/annotations/instances_val2017_100.json",
#     ),
# ]


def _is_relative_local_path(path: os.PathLike):
    path_str = os.fsdecode(path)
    return ("://" not in path_str) and not os.path.isabs(path)


def _maybe_prepend_base_path(base_path: Optional[os.PathLike], path: os.PathLike):
    """
    Prepends the provided path with a base path prefix if:
    1) base path is not None;
    2) path is a local path
    """
    if base_path is None:
        return path
    if _is_relative_local_path(path):
        return os.path.join(base_path, path)
    return path



def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def get_epfl_dicts(img_ann):
    imgs_anns = open(img_ann).read().splitlines()
    imgs_anns.sort()
    dataset_dicts = []
    records = {}
    objs = []
    for idx, _v in enumerate(imgs_anns):
        record = {}
        v = _v.split(',')
        filename = v[0]
        obj = {
        "bbox": [int(v[1]), int(v[2]), int(v[3]), int(v[4])],
        "bbox_mode": BoxMode.XYXY_ABS,
        "category_id": 0,
        "iscrowd": 0
        }
        if filename not in records:
            records[filename] = []
        records[filename].append(obj)

    for idx, v in enumerate(records.keys()):
        record = {}
        record["annotations"] = records[v]
        record["file_name"] = v
        height, width = cv2.imread(v).shape[:2]
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        dataset_dicts.append(record)

        
        
    return dataset_dicts
def register_dataset(dataset_data: DatasetInfo, datasets_root: Optional[os.PathLike] = None):
    """
    Registers provided COCO DensePose dataset

    Args:
    dataset_data: CocoDatasetInfo
        Dataset data
    datasets_root: Optional[os.PathLike]
        Datasets root folder (default: None)
    """
    annotations_fpath = _maybe_prepend_base_path(datasets_root, dataset_data.annotations_fpath)
    images_root = _maybe_prepend_base_path(datasets_root, dataset_data.images_root)

    thing_classes = []
    if 'balloon' in dataset_data.name:
        DatasetCatalog.register(dataset_data.name, lambda ir = images_root :get_balloon_dicts(ir))
        thing_classes =  ["balloon"]
    if 'epfl_relabel_eavise' in dataset_data.name:
        DatasetCatalog.register(dataset_data.name, get_balloon_dicts(annotations_fpath))
        thing_classes = ["person"]
    print(thing_classes)
    MetadataCatalog.get(dataset_data.name).set(
        image_root=images_root,
        thing_classes = thing_classes
    )


def register_datasets(
    datasets_data: Iterable[DatasetInfo], datasets_root: Optional[os.PathLike] = None
):
    """
    Registers provided COCO DensePose datasets

    Args:
    datasets_data: Iterable[CocoDatasetInfo]
        An iterable of dataset datas
    datasets_root: Optional[os.PathLike]
        Datasets root folder (default: None)
    """
    for dataset_data in datasets_data:
        register_dataset(dataset_data, datasets_root)
