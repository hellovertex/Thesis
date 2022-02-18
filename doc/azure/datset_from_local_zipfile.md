# How to create a dataset instance on Azure Machine Learning

1. Download .zip file from hsmithy
2. Upload .zip file to azure filestore
3. Connect filetore to Azure ML using Datastore
4. Create Dataset (FileDataset) from Datastore
5. Run azureml.core.Workspace.Datasets[/'my-dataset/'].mount() on target compute instance