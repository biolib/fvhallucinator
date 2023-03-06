FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

RUN rm /etc/apt/sources.list.d/cuda.list && \
    apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential \
      wget \
      && rm -rf /var/lib/apt/lists/

RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/opt/conda/bin:$PATH"
WORKDIR /home/biolib/

RUN conda create -n fvhallucinator python=3.7
SHELL ["conda", "run", "-n", "fvhallucinator", "/bin/bash", "-c"]

COPY PyRosetta4.Release.python37.linux.release-321.tar.bz2 PyRosetta4.Release.python37.linux.release-321.tar.bz2
RUN tar -xjf PyRosetta4.Release.python37.linux.release-321.tar.bz2
RUN pip install -e PyRosetta4.Release.python37.linux.release-321/setup ml-collections==0.1.0

COPY requirements.txt .

COPY data data
#COPY examples examples

RUN pip3 install -r requirements.txt

COPY src src
COPY trained_models trained_models
COPY filter.py . 
COPY generate_complexes_from_sequences.py .
COPY generate_fvs_from_sequences.py .
COPY process_designs.py .

RUN echo "source activate fvhallucinator" > ~/.bashrc
COPY hallucinate.py .
COPY hallucinate_wrapper.py .
COPY test.sh test.sh
RUN chmod +x test.sh