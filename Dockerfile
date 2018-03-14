FROM bvlc/caffe:cpu
RUN apt-get update && apt-get install -y emacs less unzip htop
WORKDIR /workspace
RUN /opt/caffe/scripts/download_model_from_gist.sh 3785162f95cd2d5fee77
RUN wget http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel
WORKDIR /opt/caffe
RUN /opt/caffe/scripts/download_model_binary.py models/bvlc_reference_caffenet
RUN /opt/caffe/data/ilsvrc12/get_ilsvrc_aux.sh
RUN pip install -r examples/web_demo/requirements.txt
RUN apt-get update && apt-get install -y iproute
COPY web_demo/app.py examples/web_demo
COPY web_demo/templates/. examples/web_demo/templates
CMD ["python", "examples/web_demo/app.py"]
