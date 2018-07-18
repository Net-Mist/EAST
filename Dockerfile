FROM tensorflow/tensorflow:1.9.0-gpu-py3

RUN apt update \
    # For open-cv
    && apt install -y libsm6 libxext6 libxrender-dev

RUN pip install opencv-python==3.4.1.15 \
    shapely==1.6.4.post1 \
    Cython==0.28.4

COPY . /root/east

WORKDIR /root/east

RUN cd src/training/gen_geo_map \
    && bash build_ext.sh \
    && cd ../../evaluating/lanms \
    && make clean \
    && make
