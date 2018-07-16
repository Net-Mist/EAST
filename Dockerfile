FROM tensorflow/tensorflow:1.9.0-gpu-py3

RUN apt update \
    # For open-cv
    && apt install -y libsm6 libxext6 libxrender-dev

RUN pip install opencv-python \
    shapely \
    cython

COPY . /root/east

WORKDIR /root/east

RUN cd training/gen_geo_map \
    && bash build_ext.sh
