class Model:
    def __init__(
        self,
        model_name,
        model_class,
        num_layers,
        ndim_out,
        activation="relu",
        dropout=0.2,
        residual=False,
        aggregation=None,
        num_neighbors=None,
        norm=False,
        trained_model=None
    ):
        self.model_name = model_name
        self.model_class = model_class
        self.num_layers = num_layers
        self.ndim_out = ndim_out
        self.activation = activation
        self.dropout = dropout
        self.residual = residual
        self.aggregation = aggregation
        self.num_neighbors = num_neighbors
        self.norm = norm
        self.trained_model = trained_model
