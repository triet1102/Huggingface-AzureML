from azureml.core.model import Model
from azureml.core import Workspace


def main(model_name,
         model_path):
    ws = Workspace.from_config()
    Model.register(ws, model_name=model_name, model_path=model_path)


if __name__ == "__main__":
    model_name = "sequence_classification"
    model_path = "onnx/model.onnx"
    main(model_name,
         model_path)
    
    exit(0)
