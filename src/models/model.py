class Model:
    def __init__(
        self,
        model_name,
        model_class,
        residual=False,
        num_neighbors=None,
        norm=False,
        trained_model=None
    ):
        self.model_name = model_name
        self.model_class = model_class
        self.residual = residual
        self.num_neighbors = num_neighbors
        self.norm = norm
        self.trained_model = trained_model
