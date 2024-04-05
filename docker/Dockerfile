FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

# Copy the local cellmincer repo
ADD . /software/cellmincer

LABEL maintainer="Brice Wang <brice@broadinstitute.org>"
ENV DOCKER=true \
    CONDA_AUTO_UPDATE_CONDA=false \
    CONDA_DIR="/opt/conda"
ENV PATH="$CONDA_DIR/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates sudo \
 && apt-get clean \
 && apt-get -y install ffmpeg \
 && sudo rm -rf /var/lib/apt/lists/* \
# get miniconda
 && curl -so $HOME/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py312_24.1.2-0-Linux-x86_64.sh \
 && chmod +x $HOME/miniconda.sh \
 && $HOME/miniconda.sh -b -p $CONDA_DIR \
 && rm $HOME/miniconda.sh \
# install cellmincer and its dependencies
 && yes | pip install -e /software/cellmincer/ \
 && conda clean -yaf \
 && sudo rm -rf ~/.cache/pip
