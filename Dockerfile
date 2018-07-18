FROM tensorflow/tensorflow:1.9.0-gpu-py3

RUN apt update \
    # For open-cv
    && apt install -y libsm6 libxext6 libxrender-dev

RUN pip install opencv-python==3.4.1.15 \
    shapely==1.6.4.post1 \
    Cython==0.28.4 \
    # For demo server
    fire==0.1.3 \
    flask==1.0.2 \
    gunicorn==19.9.0

RUN mkdir -p /root/server_log

COPY src /root/east

WORKDIR /root/east

RUN cd training/gen_geo_map \
    && bash build_ext.sh \
    && cd ../../evaluating/lanms \
    && make clean \
    && make
