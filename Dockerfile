ARG tag=20.12
ARG dev=false
FROM nvcr.io/nvidia/pytorch:${tag}-py3 AS base
ARG tag

RUN pip install nvidia-pyindex &&  pip install tritonclient[all]

ADD . /opt/exportlib

FROM base AS true
RUN pip install -e /opt/exportlib

FROM base AS false
RUN pip install /opt/exportlib && rm -rf /opt/exportlib

FROM ${dev}
