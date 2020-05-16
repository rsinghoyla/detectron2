

#from .datasets import BASE_DATASETS 
from .custom_datasets import DATASETS 
from .custom_datasets import register_datasets 

DEFAULT_DATASETS_ROOT = "datasets"


register_datasets(COCO_DATASETS, DEFAULT_DATASETS_ROOT)
#register_coco_datasets(BASE_COCO_DATASETS, DEFAULT_DATASETS_ROOT)
