SEEDS = [0, 1, 2]

TASKLIST = [
    "rna_cm", "rna_go", "rna_ligand", "rna_prot",
    "rna_if", "rna_site",
    "rna_site_redundant", "rna_cm_redundant", "rna_prot_redundant"
]

SPLITS = ["struc", "seq", "rand"]

REPRESENTATIONS = ["2.5D", "2D", "2D_GCN", "GVP", "GVP_2.5D"]

METRICS = {
    "rna_cm": "balanced_accuracy",
    "rna_go": "jaccard",
    "rna_ligand": "balanced_accuracy",
    "rna_prot": "balanced_accuracy",
    "rna_site": "balanced_accuracy",
    "rna_if": "accuracy",
}

BEST_HPARAMS = {
    'rna_cm_redundant': {
        '2.5D': {
            'struc': {
                'num_layers': 3,
                'hidden_channels': 128,
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            },
            'seq': {
                'num_layers': 3,
                'hidden_channels': 128,
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            },
            'rand': {
                'num_layers': 6,
                'hidden_channels': 128,
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        },
        '2D': {
            'struc': {
                'num_layers': 3,
                'hidden_channels': 128,
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        },
        '2D_GCN': {
            'struc': {
                'num_layers': 2,
                'hidden_channels': 128,
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        },
        'GVP': {
            'struc': {
                'num_layers': 3,
                'h_node_dim': (128, 2),
                'h_edge_dim':  (32, 1),
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        },
        'GVP_2.5D': {
            'struc': {
                'num_layers': 3,
                'h_node_dim': (32, 2),
                'h_edge_dim':  (32, 1),
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        }
    },
    'rna_cm': {
        '2.5D': {
            'struc': {
                'num_layers': 3,
                'hidden_channels': 128,
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            },
            'seq': {
                'num_layers': 3,
                'hidden_channels': 128,
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            },
            'rand': {
                'num_layers': 3,
                'hidden_channels': 128,
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        },
        '2D': {
            'struc': {
                'num_layers': 3,
                'hidden_channels': 128,
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        },
        '2D_GCN': {
            'struc':{
                'num_layers': 2,
                'hidden_channels': 128,
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        },
        'GVP': {
            'struc': {
                'num_layers': 3,
                'h_node_dim': (128, 2),
                'h_edge_dim':  (32, 1),
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        },
        'GVP_2.5D': {
            'struc': {
                'num_layers': 3,
                'h_node_dim': (32, 2),
                'h_edge_dim':  (32, 1),
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        }
    },
    'rna_prot': {
        '2.5D': {
            'struc': {
                'num_layers': 4,
                'hidden_channels': 128,
                'dropout_rate': 0.2,
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'loss_weights': 'sqrt_ratio',
            },
            'seq': {
                'num_layers': 4,
                'hidden_channels': 128,
                'dropout_rate': 0.2,
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'loss_weights': 'sqrt_ratio',
            },
            'rand': {
                'num_layers': 4,
                'hidden_channels': 128,
                'dropout_rate': 0.2,
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'loss_weights': 'sqrt_ratio',
            }
        },
        '2D': {
            'struc': {
                'num_layers': 4,
                'hidden_channels': 64,
                'dropout_rate': 0.2,
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.01,
                'loss_weights': 'sqrt_ratio',
            },
        },
        '2D_GCN': {
            'struc': {
                'num_layers': 4,
                'hidden_channels': 64,
                'dropout_rate': 0.2,
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.01,
                'loss_weights': 'sqrt_ratio',
            },
        },
        'GVP': {
            'struc': {
                'num_layers': 4,
                'h_node_dim': (32, 2),
                'h_edge_dim':  (32, 1),
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        },
        'GVP_2.5D': {
            'struc': {
                'num_layers': 4,
                'h_node_dim': (32, 2),
                'h_edge_dim':  (32, 1),
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        }
    },
    'rna_prot_redundant': {
        '2.5D':{
            'struc': {
                'num_layers': 4,
                'hidden_channels': 128,
                'dropout_rate': 0.2,
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'loss_weights': 'sqrt_ratio',
            },
            'seq': {
                'num_layers': 4,
                'hidden_channels': 128,
                'dropout_rate': 0.2,
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'loss_weights': 'sqrt_ratio',
            },
            'rand': {
                'num_layers': 6,
                'hidden_channels': 128,
                'dropout_rate': 0.2,
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'loss_weights': 'sqrt_ratio',
            }
        },
        '2D': {
            'struc': {
                'num_layers': 4,
                'hidden_channels': 64,
                'dropout_rate': 0.2,
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.01,
                'loss_weights': 'sqrt_ratio',
            },
        },
        '2D_GCN': {
            'struc': {
                'num_layers': 4,
                'hidden_channels': 64,
                'dropout_rate': 0.2,
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.01,
                'loss_weights': 'sqrt_ratio',
            },
        },
        'GVP': {
            'struc': {
                'num_layers': 4,
                'h_node_dim': (32, 2),
                'h_edge_dim':  (32, 1),
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        },
        'GVP_2.5D': {
            'struc': {
                'num_layers': 4,
                'h_node_dim': (32, 2),
                'h_edge_dim':  (32, 1),
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        }
    },
    'rna_site': {
        '2.5D': {
            'struc': {
                'num_layers': 4,
                'hidden_channels': 256,
                'batch_size': 8,
                'epochs': 40,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            },
            'seq': {
                'num_layers': 4,
                'hidden_channels': 256,
                'batch_size': 8,
                'epochs': 40,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            },
            'rand': {
                'num_layers': 4,
                'hidden_channels': 256,
                'batch_size': 8,
                'epochs': 40,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        },
        '2D': {
            'struc': {
                'num_layers': 2,
                'hidden_channels': 128,
                'batch_size': 8,
                'epochs': 40,
                'learning_rate': 0.0001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        },
        '2D_GCN': {
            'struc': {
                'num_layers': 4,
                'hidden_channels': 256,
                'batch_size': 8,
                'epochs': 100,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'ratio',
            }
        },
        'GVP': {
            'struc': {
                'num_layers': 6,
                'h_node_dim': (32, 2),
                'h_edge_dim':  (32, 1),
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        },
        'GVP_2.5D': {
            'struc': {
                'num_layers': 6,
                'h_node_dim': (32, 2),
                'h_edge_dim':  (32, 1),
                'epochs': 100,
                'batch_size': 8,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        }
    },
    'rna_site_redundant': {
        '2.5D': {
            'struc': {
                'num_layers': 4,
                'hidden_channels': 256,
                'batch_size': 8,
                'epochs': 40,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            },
            'seq': {
                'num_layers': 4,
                'hidden_channels': 256,
                'batch_size': 8,
                'epochs': 40,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            },
            'rand': {
                'num_layers': 4,
                'hidden_channels': 256,
                'batch_size': 8,
                'epochs': 40,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        },
        '2D': {
            'struc': {
                'num_layers': 2,
                'hidden_channels': 128,
                'batch_size': 8,
                'epochs': 40,
                'learning_rate': 0.0001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        },
        '2D_GCN': {
            'struc': {
                'num_layers': 2,
                'hidden_channels': 128,
                'batch_size': 8,
                'epochs': 40,
                'learning_rate': 0.0001,
                'dropout_rate': 0.5,
                'loss_weights': 'ratio',
            }
        },
        'GVP': {
            'struc': {
                'num_layers': 6,
                'h_node_dim': (32, 2),
                'h_edge_dim':  (32, 1),
                'epochs': 40,
                'batch_size': 8,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        },
        'GVP_2.5D': {
            'struc': {
                'num_layers': 6,
                'h_node_dim': (32, 2),
                'h_edge_dim':  (32, 1),
                'epochs': 100,
                'batch_size': 8,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        }
    },
    'rna_ligand': {
        '2.5D':{
            'struc': {
                'num_layers': 3,
                'hidden_channels': 64,
                'batch_size': 8,
                'epochs': 20,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            },
            'seq': {
                'num_layers': 3,
                'hidden_channels': 64,
                'batch_size': 8,
                'epochs': 20,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            },
            'rand': {
                'num_layers': 3,
                'hidden_channels': 64,
                'batch_size': 8,
                'epochs': 20,
                'learning_rate': 0.001,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        }
    },
    'rna_go': {
        '2.5D':{
            'struc': {
                'num_layers': 3,
                'hidden_channels': 128,
                'epochs': 20,
                'learning_rate': 0.001,
                'batch_size': 8,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            },
            'seq': {
                'num_layers': 3,
                'hidden_channels': 128,
                'epochs': 20,
                'learning_rate': 0.001,
                'batch_size': 8,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            },
            'rand': {
                'num_layers': 3,
                'hidden_channels': 128,
                'epochs': 20,
                'learning_rate': 0.001,
                'batch_size': 8,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        }
    },
    'rna_if': {
        '2.5D':{
            'struc': {
                'num_layers': 3,
                'hidden_channels': 128,
                'epochs': 100,
                'learning_rate': 0.0001,
                'batch_size': 8,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            },
            'seq': {
                'num_layers': 3,
                'hidden_channels': 128,
                'epochs': 100,
                'learning_rate': 0.0001,
                'batch_size': 8,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            },
            'rand': {
                'num_layers': 3,
                'hidden_channels': 128,
                'epochs': 100,
                'learning_rate': 0.0001,
                'batch_size': 8,
                'dropout_rate': 0.5,
                'loss_weights': 'sqrt_ratio',
            }
        }
    }
}