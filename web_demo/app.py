import os
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import urllib
import exifutil
import caffe
import random, string

REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
WORKSPACE_DIRNAME = '/workspace'
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

# Obtain the flask app object
app = flask.Flask(__name__)

@app.route('/')
def index():
    global REPO_DIRNAME
    return flask.render_template('index.html')

@app.route('/wait.svg', methods=['GET'])
def wait():
    response = flask.make_response(flask.render_template('wait.svg'))
    response.headers['Content-Type'] = 'image/svg+xml'
    return response

@app.route('/result.json', methods=['GET'])
def result():
    clfid = int(flask.request.args.get('c', 1)) - 1
    image = exifutil.open_oriented_im(flask.request.args.get('f'))
    result = app.clf[clfid].classify_image(image)
    data = { 'html':
             flask.render_template('_result.html', result=result,
                                   cnt=clfid+1),
             "_id": clfid+1
    }
    if len(result) > 3:
        data["timetaken"] = result[3]
    return flask.jsonify(data)

@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        if imageurl.startswith("/images/"):
            filename = os.path.dirname(os.path.abspath(__file__)) + imageurl
            logging.info('Image: %s', filename)
        else:
            filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
                        werkzeug.secure_filename(randomword(10))
            filename = os.path.join(UPLOAD_FOLDER, filename_)
            f = open(filename, "w")
            f.write(urllib.urlopen(imageurl).read())
            f.close()

        image = exifutil.open_oriented_im(filename)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'error.html', error_msg='Cannot open image from URL.')

    logging.info('Image: %s', imageurl)
    return flask.render_template('results.html', classifiers=app.clf,
                                 imagesrc=embed_image_html(image),
                                 filename=filename)


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        image = exifutil.open_oriented_im(filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'error.html', error_msg='Cannot open uploaded image.')

    return flask.render_template('results.html', classifiers=app.clf,
                                 imagesrc=embed_image_html(image),
                                 filename=filename)


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )

def randomword(length):
   return ''.join(random.choice(string.lowercase) for i in range(length))

class ImagenetClassifier(object):
    default_args = {
        'mean_file': (
            '{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)),
        'class_labels_file': (
            '{}/data/ilsvrc12/synset_words.txt'.format(REPO_DIRNAME)),
        'bet_file': (
            '{}/data/ilsvrc12/imagenet.bet.pickle'.format(REPO_DIRNAME)),
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = 256
    default_args['raw_scale'] = 255.

    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                 raw_scale, class_labels_file, bet_file, image_dim, gpu_mode):
        logging.info('Loading net and associated files...')
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.name = model_def_file
        self.net = caffe.Classifier(
            model_def_file, pretrained_model_file,
            image_dims=(image_dim, image_dim), raw_scale=raw_scale,
            mean=np.load(mean_file).mean(1).mean(1), channel_swap=(2, 1, 0)
        )

        with open(class_labels_file) as f:
            labels_df = pd.DataFrame([
                {
                    'synset_id': l.strip().split(' ')[0],
                    'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                }
                for l in f.readlines()
            ])
        self.labels = labels_df.sort_values('synset_id')['name'].values

        self.bet = cPickle.load(open(bet_file))
        # A bias to prefer children nodes in single-chain paths
        # I am setting the value to 0.1 as a quick, simple model.
        # We could use better psychological models here...
        self.bet['infogain'] -= np.array(self.bet['preferences']) * 0.1

    def name():
        return self.name

    def classify_image(self, image):
        try:
            starttime = time.time()
            scores = self.net.predict([image], oversample=True).flatten()
            endtime = time.time()

            indices = (-scores).argsort()[:5]
            predictions = self.labels[indices]

            # In addition to the prediction text, we will also produce
            # the length for the progress bar visualization.
            meta = [
                (p, '%.5f' % scores[i])
                for i, p in zip(indices, predictions)
            ]
            logging.info('result: %s', str(meta))

            # Compute expected information gain
            expected_infogain = np.dot(
                self.bet['probmat'], scores[self.bet['idmapping']])
            expected_infogain *= self.bet['infogain']

            # sort the scores
            infogain_sort = expected_infogain.argsort()[::-1]
            bet_result = [(self.bet['words'][v], '%.5f' % expected_infogain[v])
                          for v in infogain_sort[:5]]
            logging.info('bet result: %s', str(bet_result))

            return (True, meta, bet_result,
                    '%.3f' % (endtime - starttime), self.name)

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one? ({})'.format(err))

def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()

def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5123)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()
    ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})

    # Initialize classifier + warm start by forward for allocation
    app.clf = []
    clf_opts = dict(ImagenetClassifier.default_args)

    # order these by how fast they are, the slowest classifiers should be
    # at the bottom.
    models = [
        {
            'model_def_file': (
                '{}/models/bvlc_reference_caffenet/deploy.prototxt'.format(REPO_DIRNAME)),
            'pretrained_model_file': (
                '{}/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'.format(REPO_DIRNAME))
        },
        {
            'model_def_file': (
                '{}/models/bvlc_googlenet/deploy.prototxt'.format(REPO_DIRNAME)),
            'pretrained_model_file': (
                '{}/models/bvlc_googlenet/bvlc_googlenet.caffemodel'.format(REPO_DIRNAME))
        },
        {
            'model_def_file': (
                '{}/models/finetune_flickr_style/deploy.prototxt'.format(REPO_DIRNAME)),
            'pretrained_model_file': (
                '{}/models/finetune_flickr_style/finetune_flickr_style.caffemodel'.format(REPO_DIRNAME))
        },
        {
            'model_def_file': (
                '{}/models/78047f3591446d1d7b91/VGG_CNN_M_2048_deploy.prototxt'.format(WORKSPACE_DIRNAME)),
            'pretrained_model_file': (
                '{}/VGG_CNN_M_2048.caffemodel'.format(WORKSPACE_DIRNAME)),
        },
        {
            'model_def_file': (
                '{}/models/3785162f95cd2d5fee77/VGG_ILSVRC_19_layers_deploy.prototxt'.format(WORKSPACE_DIRNAME)),
            'pretrained_model_file': (
                '{}/VGG_ILSVRC_19_layers.caffemodel'.format(WORKSPACE_DIRNAME)),
        },
    ]

    for model in models:
        clf_opts.update(model)
        app.clf.append(ImagenetClassifier(**clf_opts))

    if opts.debug:
        app.run(debug=False, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
