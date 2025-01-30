"""experiment setup."""

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import BenchmarkBindingSite
from rnaglib.transforms import GraphRepresentation

# Setup task
ta_SITE_bm = BenchmarkBindingSite(root="RNA_Site_literature", debug=False, recompute=False)
ta_SITE_bm.dataset.add_representation(GraphRepresentation(framework="pyg"))
ta_SITE_bm.set_loaders(recompute=True)

# Create model
models_SITE_bm = [
    PygModel(
        ta_SITE_bm.metadata["description"]["num_node_features"],
        ta_SITE_bm.metadata["description"]["num_classes"],
        graph_level=False,
        num_layers=i,
    )
    for i in range(3)
]
