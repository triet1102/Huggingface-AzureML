# run-pytorch.py
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig

if __name__ == "__main__":
    # Init workspace
    ws = Workspace.from_config()
    
    # Init experiment
    experiment = Experiment(workspace=ws, name='bert-sequenceClassification-train')
    
    # Init configuration
    config = ScriptRunConfig(source_directory='./src',
                             script='train.py',
                             compute_target='cpu-cluster',
                             arguments=[
                                 "--checkpoint", "bert-base-uncased",
                                 "--learning-rate", 5e-5,
                                 "--batch-size", 8,
                                 "--num-epoch", 3,
                                 "--num-warmup", 0,
                                 "--save-model-dir", "bertSeqClassification_v1"
                             ])

    # Set up pytorch environment
    env = Environment.from_conda_specification(
        name='venv',
        file_path='environment.yml'
    )

    config.run_config.environment = env
    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print(f"See training insights from: \n{aml_url}")