# FROM adoptopenjdk/openjdk11:aarch64-ubuntu-jdk-11.0.10_9-slim
FROM adoptopenjdk/openjdk11:jdk-11.0.10_9-slim

# ENV TZ=Asia/Ho_Chi_Minh
# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    htop unzip wget sox bc ffmpeg net-tools \
    telnet vim build-essential python3-dev \
    procps python3-pip libopenblas-dev libopenblas-base libopenmpi-dev libomp-dev


# Miniconda
WORKDIR /
    # ADD Miniconda3-py310_25.1.1-2-Linux-aarch64.sh /opt
    # ENV MINICONDA_INSTALL_FILENAME="Miniconda3-py310_25.1.1-2-Linux-aarch64.sh"
    ADD Miniconda3-py310_25.1.1-2-Linux-x86_64.sh /opt
    ENV MINICONDA_INSTALL_FILENAME="Miniconda3-py310_25.1.1-2-Linux-x86_64.sh"

    RUN chmod +x "/opt/${MINICONDA_INSTALL_FILENAME}"

    ENV MINICONDA_ROOT="/opt/Miniconda3"
    RUN rm -rf "${MINICONDA_ROOT}"
    RUN "/opt/${MINICONDA_INSTALL_FILENAME}" -b -p "${MINICONDA_ROOT}"

    ENV PATH="${MINICONDA_ROOT}/bin:${PATH}"
    RUN ln -sf "${MINICONDA_ROOT}/bin/python" "/usr/bin/python3" && \
        ln -sf "${MINICONDA_ROOT}/bin/python" "/usr/bin/python" && \
        ln -sf "${MINICONDA_ROOT}/bin/pip" "/usr/bin/pip3" && \
        ln -sf "${MINICONDA_ROOT}/bin/pip" "/usr/bin/pip"

    # install python requirements
    ADD ./requirements.txt /opt
    RUN /usr/bin/pip3 install wheel

    # ADD ./packages/torch-2.3.0+rocm5.7-cp310-cp310-linux_x86_64.whl /opt
    # RUN /usr/bin/pip3 install /opt/torch-2.3.0+rocm5.7-cp310-cp310-linux_x86_64.whl
    RUN /usr/bin/pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
    # ADD ./packages/torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl /opt
    # RUN /usr/bin/pip3 install /opt/torchaudio-2.3.0-cp310-cp310-linux_x86_64.whl

    RUN /usr/bin/pip3 install -r /opt/requirements.txt  
    RUN /usr/bin/pip3 install -I numpy==1.26.4 

RUN mkdir -p /opt/server_e2e
ADD streaming_decoder /opt/server_e2e/streaming_decoder

ADD Read_Number_Py    /opt/Read_Number_Py
ADD run_service.sh    /opt/server_e2e

RUN chmod +x /opt/server_e2e/run_service.sh
ENTRYPOINT [ "/opt/server_e2e/run_service.sh" ]
