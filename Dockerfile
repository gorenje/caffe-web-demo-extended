# Using the existing Docker image as a starting point.
FROM bvlc/caffe:cpu
RUN apt-get update && apt-get install -y emacs less unzip htop
WORKDIR /workspace

# Retrieve the VGG model
RUN /opt/caffe/scripts/download_model_from_gist.sh 3785162f95cd2d5fee77
RUN wget http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel

# Setup the default model, install the ilsvrc auxilary data and install
# python libraries for he web demo code
WORKDIR /opt/caffe
RUN /opt/caffe/scripts/download_model_binary.py models/bvlc_reference_caffenet
RUN /opt/caffe/data/ilsvrc12/get_ilsvrc_aux.sh
RUN pip install -r examples/web_demo/requirements.txt

# replace the original caffe code.
COPY web_demo/app.py examples/web_demo
COPY web_demo/templates/. examples/web_demo/templates

# start web server
CMD ["python", "examples/web_demo/app.py"]
