from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import os

import tensorflow as tf

from medium_show_and_tell_caption_generator.caption_generator import CaptionGenerator
from medium_show_and_tell_caption_generator.model import ShowAndTellModel
from medium_show_and_tell_caption_generator.vocabulary import Vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("model_path", "../model/show-and-tell.pb", "Model graph def path")
tf.flags.DEFINE_string("vocab_file", "../etc/word_counts.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "../imgs/",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def main(_):
    model = ShowAndTellModel(FLAGS.model_path)
    vocab = Vocabulary(FLAGS.vocab_file)
    filenames = _load_filenames()

    generator = CaptionGenerator(model, vocab)
    with open('../pred.txt', 'w') as f1:
        for filename in filenames:
            with tf.gfile.GFile(filename, "rb") as f:
                image = f.read()
            captions = generator.beam_search(image)
            print("Captions for image %s:" % os.path.basename(filename))
            for i, caption in enumerate(captions):
                # Ignore begin and end tokens <S> and </S>.
                sentence = [vocab.id_to_token(w) for w in caption.sentence[1:-1]]
                sentence = " ".join(sentence)
                if i == 1:
                    f1.write("%s \n" % sentence)
                    # print("this is---", sentence)
                print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))


def _load_filenames():
    filenames = []
    fn =[]
    directory = FLAGS.input_files
    for filename in os.listdir(directory):
        if filename.endswith(".JPG") or filename.endswith(".png") or filename.endswith(".jpg"):
            pathname = os.path.join(directory, filename)
            filenames.append(pathname)
            print(filenames)
            continue
        else:
            continue
    return filenames


if __name__ == "__main__":
    tf.app.run()
