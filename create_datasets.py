from constants import TASKLIST

from rnaglib.tasks import get_task

from utils import CustomLigandIdentification

def create_task_dataset(task):
    # Instantiate task
    remove_redundancy = not task.endswith("redundant")
    task_args = {
        "root": f"roots/{task}",
        "redundancy_removal": remove_redundancy,
        "precomputed": remove_redundancy,
    }
    if task != "rna_ligand":
        ta = get_task(task_id=task.split("_redundant")[0], **task_args)
    else:
        ta = CustomLigandIdentification(**task_args)

if __name__ == "__main__":
    for task in TASKLIST:
        create_task_dataset(task)