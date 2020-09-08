FROM arm32v7/debian:buster

MAINTAINER Mahmoud Elsousy

RUN mkdir /emad

WORKDIR /emad

RUN apt update

RUN apt install -y wget bzip2

RUN wget https://github.com/jjhelmus/berryconda/releases/download/v2.0.0/Berryconda3-2.0.0-Linux-armv7l.sh

RUN chmod +x Berryconda3-2.0.0-Linux-armv7l.sh

RUN ./Berryconda3-2.0.0-Linux-armv7l.sh -b

RUN rm Berryconda3-2.0.0-Linux-armv7l.sh

ENV PATH="/root/berryconda3/bin:${PATH}"

RUN echo "export PATH=/root/berryconda3/bin:${PATH}" >> /root/.profile

RUN conda update -y conda

RUN conda update -y pip

RUN pip install --upgrade  pip

RUN conda install -y -c numba numba

RUN conda install -y scikit-learn matplotlib pandas jupyterlab

RUN pip install pyod==0.8.1 -i https://www.piwheels.org/simple

RUN pip install nose -i https://www.piwheels.org/simple

COPY Brotli-1.0.7-cp36-cp36m-linux_armv7l.whl Brotli-1.0.7-cp36-cp36m-linux_armv7l.whl

RUN pip install Brotli-1.0.7-cp36-cp36m-linux_armv7l.whl

RUN pip install dash

RUN pip install dash-bootstrap-components

RUN conda clean -tipsy \
    && find /root/berryconda3/ -type f,l -name '*.a' -delete \
    && find /root/berryconda3/ -type f,l -name '*.pyc' -delete \
    && find /root/berryconda3/ -type f,l -name '*.js.map' -delete \
    && rm -rf /root/berryconda3/pkgs

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

ADD EmAD_Dash /emad/EmAD_Dash

EXPOSE 4444

EXPOSE 9999

ENTRYPOINT ["python", "/emad/EmAD_Dash/appfull.py"]

# ENTRYPOINT ["jupyter-lab", "--ip=*", "--allow-root", "--no-browser"]
