FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04:20220208.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/pytorch-1.10

# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8 \
    pip=20.2.4 \
    pytorch=1.10.0 \
    torchvision=0.11.1 \
    torchaudio=0.10.0 \
    cudatoolkit=11.1.1 \
    nvidia-apex=0.1.0 \
    gxx_linux-64 \
    -c anaconda -c pytorch -c conda-forge

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install pip dependencies
RUN pip install 'matplotlib>=3.3,<3.4' \
                'psutil>=5.8,<5.9' \
                'tqdm>=4.59,<4.63' \
                'pandas>=1.3,<1.4' \
                'scipy>=1.5,<1.8' \
                'numpy>=1.10,<1.22' \
                'ipykernel~=6.0' \
                'azureml-core==1.38.0.post2' \
                'azureml-defaults==1.38.0' \
                'azureml-mlflow==1.38.0' \
                'azureml-telemetry==1.38.0' \
                'tensorboard==2.6.0' \
                'tensorflow-gpu==2.6.0' \
                'onnxruntime-gpu>=1.7,<1.10' \
                'horovod==0.23' \
                'future==0.18.2' \
                'torch-tb-profiler==0.3.1'


# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
