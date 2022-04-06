from azureml.core import Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import LocalWebservice
from azureml.core.model import Model
from azureml.core import Workspace
import requests
import json

ws = Workspace.from_config()
model = Model(ws, model_name="BertSentenceClassification", id="BertSentenceClassification:1")

env = Environment(name="project_environment")
dummy_inference_config = InferenceConfig(
    environment=env,
    source_directory="./src",
    entry_script="./echo_score.py",
)

deployment_config = LocalWebservice.deploy_configuration(port=6789)


service = Model.deploy(
    ws,
    "myservice",
    [model],
    dummy_inference_config,
    deployment_config,
    overwrite=True,
)
service.wait_for_deployment(show_output=True)


uri = service.scoring_uri
requests.get("http://localhost:6789")
headers = {"Content-Type": "application/json"}
data = {
    "query": "What color is the fox",
    "context": "The quick brown fox jumped over the lazy dog.",
}
data = json.dumps(data)
response = requests.post(uri, data=data, headers=headers)
print(response.json())