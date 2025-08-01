import random
import torch
import numpy as np

from rnaglib.tasks import LigandIdentification
from rnaglib.tasks.RNA_Ligand.prepare_dataset import PrepareDataset
from rnaglib.dataset_transforms import CDHitComputer, StructureDistanceComputer

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class CustomLigandIdentification(LigandIdentification):
        def __init__(self, redundancy_removal, **kwargs):
                self.redundancy_removal = redundancy_removal
                super().__init__(**kwargs)
        def post_process(self):
                """The task-specific post processing steps to remove redundancy and compute distances which will be used by the splitters.
                """
                cd_hit_computer = CDHitComputer(similarity_threshold=0.9)
                if self.redundancy_removal:
                        prepare_dataset = PrepareDataset(distance_name="cd_hit", threshold=0.9)
                us_align_computer = StructureDistanceComputer(name="USalign")
                self.dataset = cd_hit_computer(self.dataset)
                if self.redundancy_removal:
                        self.dataset = prepare_dataset(self.dataset)
                self.dataset = us_align_computer(self.dataset) 