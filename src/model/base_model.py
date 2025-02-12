class BaseModel:
    def __init__(self, model_name: str, hf_token: str = None, **kwargs):
        self.model_name = model_name
        self.hf_token = hf_token
        self.kwargs = kwargs

    def _initialization(self):
        pass

    def _load_model(self):
        pass
