from __future__ import division
from __future__ import print_function
import codecs
import sys

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from DataLoader import FilePaths


class DecoderType:
    BestPath = 0
    WordBeamSearch = 1
    BeamSearch = 2


class Model:
    batchSize = 10  # 50
    imgSize = (800, 64)
    maxTextLen = 100

    def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False):
        self.charList = charList
        self.decoderType = decoderType
        self.mustRestore = mustRestore
        self.snapID = 0
        self.inputImgs = tf.compat.v1.placeholder(tf.float32, shape=(
            None, Model.imgSize[0], Model.imgSize[1]))
        self.setupCNN()
        self.setupRNN()
        self.setupCTC()

        self.batchesTrained = 0
        self.learningRate = tf.compat.v1.placeholder(tf.float32, shape=[])
        self.optimizer = tf.compat.v1.train.RMSPropOptimizer(
            self.learningRate).minimize(self.loss)

        (self.sess, self.saver) = self.setupTF()

        self.training_loss_summary = tf.compat.v1.summary.scalar('loss', self.loss)
        self.writer = tf.compat.v1.summary.FileWriter(
            './logs', self.sess.graph)
        self.merge = tf.compat.v1.summary.merge(
            [self.training_loss_summary])

    def setupCNN(self):

        cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3)

        # First Layer: Conv (5x5) + Pool (2x2) - Output size: 400 x 32 x 64
        with tf.compat.v1.name_scope('Conv_Pool_1'):
            kernel = tf.Variable(
                tf.random.truncated_normal([5, 5, 1, 64], stddev=0.1))
            conv = tf.nn.conv2d(
                cnnIn4d, kernel, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)
            pool = tf.nn.max_pool2d(learelu, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

        # Second Layer: Conv (5x5) + Pool (1x2) - Output size: 400 x 16 x 128
        with tf.compat.v1.name_scope('Conv_Pool_2'):
            kernel = tf.Variable(tf.random.truncated_normal(
                [5, 5, 64, 128], stddev=0.1))
            conv = tf.nn.conv2d(
                pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)
            pool = tf.nn.max_pool2d(learelu, (1, 1, 2, 1), (1, 1, 2, 1), 'VALID')

        # Third Layer: Conv (3x3) + Pool (2x2) + Simple Batch Norm - Output size: 200 x 8 x 128
        with tf.compat.v1.name_scope('Conv_Pool_BN_3'):
            kernel = tf.Variable(tf.random.truncated_normal(
                [3, 3, 128, 128], stddev=0.1))
            conv = tf.nn.conv2d(
                pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            mean, variance = tf.nn.moments(conv, axes=[0])
            batch_norm = tf.nn.batch_normalization(
                conv, mean, variance, offset=None, scale=None, variance_epsilon=0.001)
            learelu = tf.nn.leaky_relu(batch_norm, alpha=0.01)
            pool = tf.nn.max_pool2d(learelu, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

        # Fourth Layer: Conv (3x3) - Output size: 200 x 8 x 256
        with tf.compat.v1.name_scope('Conv_4'):
            kernel = tf.Variable(tf.random.truncated_normal(
                [3, 3, 128, 256], stddev=0.1))
            conv = tf.nn.conv2d(
                pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)

        # Fifth Layer: Conv (3x3) + Pool(2x2) - Output size: 100 x 4 x 256
        with tf.compat.v1.name_scope('Conv_Pool_5'):
            kernel = tf.Variable(tf.random.truncated_normal(
                [3, 3, 256, 256], stddev=0.1))
            conv = tf.nn.conv2d(
                learelu, kernel, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)
            pool = tf.nn.max_pool2d(learelu, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

        # Sixth Layer: Conv (3x3) + Pool(1x2) + Simple Batch Norm - Output size: 100 x 2 x 512
        with tf.compat.v1.name_scope('Conv_Pool_BN_6'):
            kernel = tf.Variable(tf.random.truncated_normal(
                [3, 3, 256, 512], stddev=0.1))
            conv = tf.nn.conv2d(
                pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            mean, variance = tf.nn.moments(conv, axes=[0])
            batch_norm = tf.nn.batch_normalization(
                conv, mean, variance, offset=None, scale=None, variance_epsilon=0.001)
            learelu = tf.nn.leaky_relu(batch_norm, alpha=0.01)
            pool = tf.nn.max_pool2d(learelu, (1, 1, 2, 1), (1, 1, 2, 1), 'VALID')

        # Seventh Layer: Conv (3x3) + Pool (1x2) - Output size: 100 x 1 x 512
        with tf.compat.v1.name_scope('Conv_Pool_7'):
            kernel = tf.Variable(tf.random.truncated_normal(
                [3, 3, 512, 512], stddev=0.1))
            conv = tf.nn.conv2d(
                pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)
            pool = tf.nn.max_pool2d(learelu, (1, 1, 2, 1), (1, 1, 2, 1), 'VALID')

            self.cnnOut4d = pool

    def setupRNN(self):
        rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])
        numHidden = 512
        cells = [tf.compat.v1.nn.rnn_cell.LSTMCell(
            num_units=numHidden, state_is_tuple=True, name='basic_lstm_cell') for _ in range(2)]
        stacked = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        ((forward, backward), _) = tf.compat.v1.nn.bidirectional_dynamic_rnn(
            cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)
        concat = tf.expand_dims(tf.concat([forward, backward], 2), 2)
        kernel = tf.Variable(tf.random.truncated_normal(
            [1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
        self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(
            value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])

    def setupCTC(self):
        self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])
        with tf.compat.v1.name_scope('CTC_Loss'):
            self.gtTexts = tf.SparseTensor(tf.compat.v1.placeholder(tf.int64, shape=[
                                           None, 2]), tf.compat.v1.placeholder(tf.int32, [None]), tf.compat.v1.placeholder(tf.int64, [2]))
            self.seqLen = tf.compat.v1.placeholder(tf.int32, [None])
            self.loss = tf.reduce_mean(tf.compat.v1.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen,
                                                      ctc_merge_repeated=True, ignore_longer_outputs_than_inputs=True))
        with tf.compat.v1.name_scope('CTC_Decoder'):

            if self.decoderType == DecoderType.BestPath:
                self.decoder = tf.nn.ctc_greedy_decoder(
                    inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
            elif self.decoderType == DecoderType.BeamSearch:
                self.decoder = tf.compat.v1.nn.ctc_beam_search_decoder(
                    inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50, merge_repeated=True)
            elif self.decoderType == DecoderType.WordBeamSearch:
                word_beam_search_module = tf.load_op_library(
                    './TFWordBeamSearch.so')
                chars = codecs.open(FilePaths.wordCharList.txt, 'r').read()
                wordChars = codecs.open(
                    FilePaths.fnWordCharList, 'r').read()
                corpus = codecs.open(FilePaths.corpus.txt, 'r').read()
                self.decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(
                    self.ctcIn3dTBC, dim=2), 25, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))
        return self.loss, self.decoder

    def setupTF(self):
        print('Python: ' + sys.version)
        print('Tensorflow: ' + tf.__version__)
        sess = tf.compat.v1.Session()
        saver = tf.compat.v1.train.Saver(max_to_keep=3)
        modelDir = '../model/'
        latestSnapshot = tf.train.latest_checkpoint(
            modelDir)
        if self.mustRestore and not latestSnapshot:
            raise Exception('No saved model found in: ' + modelDir)
        # Load saved model if available
        if latestSnapshot:
            print('Init with stored values from ' + latestSnapshot)
            saver.restore(sess, latestSnapshot)
        else:
            print('Init with new values')
            sess.run(tf.global_variables_initializer())

        return (sess, saver)

    def toSpare(self, texts):
        indices = []
        values = []
        shape = [len(texts), 0]
        for (batchElement, texts) in enumerate(texts):
            print(texts)
            labelStr = []
            for c in texts:
                print(c, '|', end='')
                labelStr.append(self.charList.index(c))
            print(' ')
            labelStr = [self.charList.index(c) for c in texts]
            if len(labelStr) > shape[1]:
                shape[1] = len(labelStr)
            for (i, label) in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)

        return (indices, values, shape)

    def decoderOutputToText(self, ctcOutput):
        encodedLabelStrs = [[] for i in range(Model.batchSize)]
        if self.decoderType == DecoderType.WordBeamSearch:
            blank = len(self.charList)
            for b in range(Model.batchSize):
                for label in ctcOutput[b]:
                    if label == blank:
                        break
                    encodedLabelStrs[b].append(label)
        else:
            decoded = ctcOutput[0][0]
            idxDict = {b: [] for b in range(Model.batchSize)}
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batchElement = idx2d[0]
                encodedLabelStrs[batchElement].append(label)
        return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]

    def trainBatch(self, batch, batchNum):
        sparse = self.toSpare(batch.gtTexts)
        rate = 0.001
        evalList = [self.merge, self.optimizer, self.loss]
        feedDict = {self.inputImgs: batch.imgs, self.gtTexts: sparse, self.seqLen: [
            Model.maxTextLen] * Model.batchSize, self.learningRate: rate}
        (loss_summary, _, lossVal) = self.sess.run(evalList, feedDict)
        self.writer.add_summary(loss_summary, batchNum)
        self.batchesTrained += 1
        return lossVal

    def return_rnn_out(self, batch, write_on_csv=False):
        numBatchElements = len(batch.imgs)
        decoded, rnnOutput = self.sess.run([self.decoder, self.ctcIn3dTBC],
                                           {self.inputImgs: batch.imgs, self.seqLen: [Model.maxTextLen] * numBatchElements})

        decoded = rnnOutput
        print(decoded.shape)

        if write_on_csv:
            s = rnnOutput.shape
            b = 0
            csv = ''
            for t in range(s[0]):
                for c in range(s[2]):
                    csv += str(rnnOutput[t, b, c]) + ';'
                csv += '\n'
            open('mat_0.csv', 'w').write(csv)

        return decoded[:, 0, :].reshape(100, 80)

    def inferBatch(self, batch):
        numBatchElements = len(batch.imgs)
        feedDict = {self.inputImgs: batch.imgs, self.seqLen: [
            Model.maxTextLen] * numBatchElements}
        evalRes = self.sess.run([self.decoder, self.ctcIn3dTBC], feedDict)
        decoded = evalRes[0]
        texts = self.decoderOutputToText(decoded)
        return texts

    def save(self):
        self.snapID += 1
        self.saver.save(self.sess, '../model/snapshot',
                        global_step=self.snapID)
