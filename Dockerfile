# Using the existing Docker image as a starting point.
FROM bvlc/caffe:cpu
RUN apt-get update && apt-get install -y emacs less unzip htop
WORKDIR /workspace

# Retrieve the VGG models
RUN /opt/caffe/scripts/download_model_from_gist.sh 3785162f95cd2d5fee77
RUN wget http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel

RUN /opt/caffe/scripts/download_model_from_gist.sh 78047f3591446d1d7b91
RUN wget http://www.robots.ox.ac.uk/%7Evgg/software/deep_eval/releases/bvlc/VGG_CNN_M_2048.caffemodel

# Setup the models we'll be running, install the ilsvrc auxilary data and
# install python libraries for he web demo code
WORKDIR /opt/caffe
RUN /opt/caffe/scripts/download_model_binary.py models/bvlc_reference_caffenet
RUN /opt/caffe/data/ilsvrc12/get_ilsvrc_aux.sh
RUN /opt/caffe/scripts/download_model_binary.py models/bvlc_googlenet
RUN /opt/caffe/scripts/download_model_binary.py models/finetune_flickr_style
RUN pip install -r examples/web_demo/requirements.txt

# replace the original web demo code.
COPY web_demo/app.py examples/web_demo
COPY web_demo/templates/. examples/web_demo/templates

# add some test images
RUN mkdir examples/web_demo/images
COPY images/. examples/web_demo/images/

# start web server
CMD ["python", "examples/web_demo/app.py"]
