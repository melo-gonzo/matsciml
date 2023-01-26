FROM vault.habana.ai/gaudi-docker/1.7.1/ubuntu20.04/habanalabs/pytorch-installer-1.13.0:latest

RUN git clone -b habana --depth 1 https://github.com/melo-gonzo/matsciml.git

WORKDIR /matsciml

RUN pip install -e .

CMD [ "python", "examples/gala_habana.py"]

