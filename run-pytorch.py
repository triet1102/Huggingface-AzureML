# run-pytorch.py
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core.runconfig import PyTorchConfiguration
import os

if __name__ == "__main__":
    os.chdir("/home/azureuser/cloudfiles/code/Users/triet.tran/huggingface")

    # Init workspace
    ws = Workspace.from_config()

    # Init experiment
    experiment = Experiment(
        workspace=ws, name='bert-sequenceClassification-train')
    distr_config = PyTorchConfiguration(process_count=8, node_count=2)

    # Init configuration
    config = ScriptRunConfig(source_directory='./src',
                             script='train.py',
                             compute_target='gpu-cluster',
                             distributed_job_config=distr_config,
                             arguments=[
                                 "--checkpoint", "bert-base-uncased",
                                 "--learning-rate", 5e-5,
                                 "--batch-size", 8,
                                 "--num-epoch", 3,
                                 "--num-warmup", 0,
                                 "--save-model", True
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
    run.wait_for_completion()

    if config.arguments[-1]:
        os.makedirs('./model', exist_ok=True)
        run.download_file(name='outputs/model.pt',
                          output_file_path='./model/model.pt'),
