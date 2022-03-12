import azureml.core
from azureml.core import Workspace, Datastore

config = {
    "subscription_id": "0d263aec-21a1-4c68-90f8-687d99ccb93b",
    "resource_group": "thesis",
    "workspace_name": "generate-train-data"
}
# connect to get-train-data workspace
ws = Workspace.get(name=config["workspace_name"],
                   subscription_id=config["subscription_id"],
                   resource_group=config["resource_group"])

file_datastore_name = 'azfilesharesdk'  # Name of the datastore to workspace
file_share_name = "azureml-filestore-d26b4c3c-4c47-455d-a485-fcf19619aaa2"  # Name of Azure file share container
account_name = "generatetraind4882846949"  # Storage account name
account_key = "qw+yXrFzCR+AuSakXHpbKEaTUvj+EWkdC3YTLbNKUpmaELeVonc78Xqpk9lTCofhx57nTYo5YjkWU4oiWHI+dQ=="  # Storage account access key

file_datastore = Datastore.register_azure_file_share(workspace=ws,
                                                     datastore_name=file_datastore_name,
                                                     file_share_name=file_share_name,
                                                     account_name=account_name,
                                                     account_key=account_key)
print(file_datastore)