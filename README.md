# rnaglib-experiments

This repository contains all code used for our experiments conducted with the `rnaglib` task suite. For full documentation please refer to [rnaglib.org](rnaglib.org) and [github.com/cgoliver/rnaglib](github.com/cgoliver/rnaglib).

# Reproducing experiments

This repository provides the necessary code to reproduce the three main experiments reported in our preprint: number of layers ablation study, splitting strategy ablation study and representation ablation study

* Prior to running any of the three experiments detailed below, you need to create the datasets. To do so, please run `python create_datasets.py`. This will create the datasets relevant to each of the tasks and store them in a folder named `roots`

* To reproduce the number of layers ablation study, run `python run_exp_nb_layers.py`. Once this is done, JSON files containing the training data will be stored in `results`. Then, you will be able to run `python plotting_scripts/make_plot_nb_layers.py` which will create (if default parameters are kept) a file named `plotting_scripts/nb_layers_ablation_2.5D.pdf` corresponding to Figure 4a of our preprint. Please note that you can tune some parameters, for instance `representation` in order to visualize the impact of the number of layers when using a different representation than 2.5D. In this case, you need to change accordingly the parameters in `run_exp_nb_layers.py` and in `make_plot_nb_layers.py`

* To reproduce the splitting strategy ablation study, run `python run_exp_splitting.py`, which will train models and dump the relevant JSONs in `results`. In order to reproduce the associated plot, run `python plotting_scripts/make_plot_splitting.py`. This will create a file named `plotting_scripts/splitting_ablation.pdf` reproducing Figure 3a of our preprint.

* To reproduce the representation ablation study, run `python run_exp_representations.py`, which will train models and dump the relevant JSONs in `results`. In order to reproduce the associated plot, run `python plotting_scripts/make_plot_representations.py`. This will create a file named `plotting_scripts/representation_ablation.pdf` reproducing Figure 4b of our preprint.

* To reproduce the benchmark table , run `python run_exp_splitting.py`, which will train each model with its default splitting and 2.5D representations with its best hyperparameters. Then run `python plotting_scripts/make_table_benchmark.py`. This will create a file named `plotting_scripts/final_benchmark.pdf` reproducing Table 2 of our preprint, alongside a CSV file `plotting_scripts/final_benchmark.csv`.

Once a training has been made in specific conditions, it won't be re-run if a subsequently used script needs to run it, unless you change the `retrain` parameter to `True`. Therefore, running experiments which trainings partially overlap won't lead to a waste of time.