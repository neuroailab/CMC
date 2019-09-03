import argparse
import pdb
from models.resnet import ResNetV2
import torch
import torch.backends.cudnn as cudnn
import tensorflow as tf
from PIL import Image
import numpy as np
from scipy.misc import imresize
from dataset import RGB2Lab
import os
from tqdm import tqdm
import torch.nn as nn
try:
    import cPickle
    pickle = cPickle
except:
    import pickle

ORIGINAL_SHAPE = [256, 256, 3]
INPUT_SHAPE = [224, 224 ,3]
TFR_PAT = 'tfrecords'


def get_parser():
    parser = argparse.ArgumentParser(
            description='Generate outputs for neural fitting')

    parser.add_argument(
            '--data', type=str, 
            default='/mnt/fs0/datasets/neural_data'\
                    + '/img_split/V4IT/tf_records/images',
            help='path to stimuli')
    parser.add_argument(
            '--model_ckpt', type=str, 
            default='/mnt/fs4/chengxuz/cmc_models/resnet50v2.pth',
            help='path to model')
    parser.add_argument(
            '--batch_size', default=32, type=int,
            help='mini-batch size')
    parser.add_argument(
            '--model', type=str, 
            default='resnet50', choices=['resnet18', 'resnet50', 'resnet101'])
    parser.add_argument(
            '--save_path', type=str,
            default='/mnt/fs4/chengxuz/v4it_temp_results/cmc_nf/V4IT_split_0',
            help='path for storing results')
    parser.add_argument(
            '--dataset_type', type=str, default='hvm')
    parser.add_argument(
            '--is_half', action='store_true', 
            help='Half the network for resnet')
    return parser


def load_model(args):
    model = ResNetV2(args.model, is_half=args.is_half)
    checkpoint = torch.load(args.model_ckpt)
    model.load_state_dict(checkpoint['model'])

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    # freeze the layers
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def forward(x, model):
    ret_output = []
    all_modules = list(model.children())[:-1]
    for each_m in all_modules[:4]:
        x = each_m(x)
    ret_output.append(x) # pool1
    for each_m in all_modules[4:]:
        if not isinstance(each_m, nn.Sequential):
            continue
        for each_m_child in each_m.children():
            x = each_m_child(x)
            ret_output.append(x)
    return ret_output


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_tfr_files(data_path):
    # Get tfrecord files
    all_tfrs_path = os.listdir(data_path)
    all_tfrs_path = filter(lambda x:TFR_PAT in x, all_tfrs_path)
    all_tfrs_path.sort()
    all_tfrs_path = [os.path.join(data_path, each_tfr) \
            for each_tfr in all_tfrs_path]

    return all_tfrs_path


_RGB2Lab = RGB2Lab()
LAB_MEAN = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
LAB_STD = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
def tolab_normalize(img):
    img = _RGB2Lab(img)
    img -= LAB_MEAN
    img /= LAB_STD
    return img


def get_one_image(string_record):
    example = tf.train.Example()
    example.ParseFromString(string_record)
    img_string = (example.features.feature['images']
                                  .bytes_list
                                  .value[0])
    img_array = np.fromstring(img_string, dtype=np.float32)
    img_array = img_array.reshape(ORIGINAL_SHAPE)
    img_array *= 255
    img_array = img_array.astype(np.uint8)
    img_array = imresize(img_array, INPUT_SHAPE)
    img_array = tolab_normalize(img_array)
    img_array = np.transpose(img_array, [2, 0, 1])
    img_array = img_array.astype(np.float32)
    return img_array


def get_batches(all_records):
    all_images = []
    for string_record in all_records:
        all_images.append(get_one_image(string_record))
    all_images = np.stack(all_images, axis=0)
    return all_images


def get_all_images(tfr_path):
    record_iterator = tf.python_io.tf_record_iterator(path=tfr_path)
    all_records = list(record_iterator)
    num_imgs = len(all_records)
    all_images = get_batches(all_records)
    return num_imgs, all_images


class NfOutput(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cpu")
        self.make_meta = True
        self.model = load_model(args)
        self.result_keys = ['only_l', 'only_ab', 'l_ab']
        self.all_writers = None

    def _get_outputs(self, curr_batch):
        args = self.args
        input_var = torch.autograd.Variable(
                torch.from_numpy(curr_batch).cuda())
        l_img, ab_img = torch.split(input_var, [1, 2], dim=1)
        l_outputs = forward(l_img, self.model.l_to_ab)
        ab_outputs = forward(ab_img, self.model.ab_to_l)

        def _transfer_outputs(outputs):
            outputs = [
                    np.asarray(output.float().to(self.device)) \
                    for output in outputs]
            outputs = [np.transpose(output, [0, 2, 3, 1]) for output in outputs]
            return outputs
        l_outputs = _transfer_outputs(l_outputs)
        ab_outputs = _transfer_outputs(ab_outputs)
        lab_outputs = [
                np.concatenate([_l_output, _ab_output], axis=-1) \
                for _l_output, _ab_output in zip(l_outputs, ab_outputs)]
        return l_outputs, ab_outputs, lab_outputs

    def _make_meta(self, all_outputs):
        args = self.args
        
        for result_key, curr_outputs in zip(self.result_keys, all_outputs):
            for save_key in self.save_keys:
                curr_folder = os.path.join(args.save_path, result_key, save_key)
                os.system('mkdir -p %s' % curr_folder)

            for save_key, curr_output in zip(self.save_keys, curr_outputs):
                curr_meta = {
                        save_key: {
                            'dtype': tf.string, 
                            'shape': (), 
                            'raw_shape': tuple(curr_output.shape[1:]),
                            'raw_dtype': tf.float32,
                            }
                        }
                meta_path = os.path.join(
                        args.save_path, result_key, 
                        save_key, 'meta.pkl')
                pickle.dump(curr_meta, open(meta_path, 'w'))
        self.make_meta = False

    def _make_writers(self, tfr_path):
        args = self.args
        
        all_writers = {}
        for result_key in self.result_keys:
            curr_writers = []
            for save_key in self.save_keys:
                write_path = os.path.join(
                        args.save_path, result_key, save_key,
                        os.path.basename(tfr_path))
                writer = tf.python_io.TFRecordWriter(write_path)
                curr_writers.append(writer)
            all_writers[result_key] = curr_writers
        self.all_writers = all_writers

    def _write_outputs(self, all_outputs):
        for result_key, curr_outputs in zip(self.result_keys, all_outputs):
            for writer, curr_output, save_key in \
                    zip(self.all_writers[result_key], 
                        curr_outputs, self.save_keys):
                for idx in range(curr_output.shape[0]):
                    curr_value = curr_output[idx]
                    save_feature = {
                            save_key: _bytes_feature(curr_value.tostring())
                            }
                    example = tf.train.Example(
                            features=tf.train.Features(feature=save_feature))
                    writer.write(example.SerializeToString())

    def _close_writers(self):
        for result_key in self.result_keys:
            for each_writer in self.all_writers[result_key]:
                each_writer.close()
        self.all_writers = None

    def write_outputs_for_one_tfr(self, tfr_path):
        args = self.args
        if args.dataset_type == 'v1_tc':
            global ORIGINAL_SHAPE
            global INPUT_SHAPE
            ORIGINAL_SHAPE = [80, 80, 3]
            INPUT_SHAPE = [40, 40, 3]
        num_imgs, all_images = get_all_images(tfr_path)

        for start_idx in range(0, num_imgs, args.batch_size):
            curr_batch = all_images[start_idx : start_idx + args.batch_size]
            all_outputs = self._get_outputs(curr_batch)

            if self.make_meta:
                self.save_keys = [
                        'conv%i' % idx for idx in range(len(all_outputs[0]))]
                self._make_meta(all_outputs)
            if self.all_writers is None:
                self._make_writers(tfr_path)
            self._write_outputs(all_outputs)
        self._close_writers()


def main():
    parser = get_parser()
    args = parser.parse_args()

    #model = load_model(args)
    if args.dataset_type == 'v1_tc':
        global TFR_PAT
        TFR_PAT = 'split'
    all_tfr_path = get_tfr_files(args.data)

    nf_output = NfOutput(args)
    
    for tfr_path in tqdm(all_tfr_path):
        nf_output.write_outputs_for_one_tfr(tfr_path)


if __name__ == '__main__':
    main()
