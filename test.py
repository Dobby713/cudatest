import os
from torchvision import transforms
from PIL import Image
import torch
from glob import glob
from options.test_options import TestOptions
from models.pix2pix_classifier_model import make_unet_generator
import matplotlib.pyplot as plt

class ProcessImage:
    def __init__(self):
        transform_list = [transforms.ToTensor(), transforms.Resize(256), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        self.transform = transforms.Compose(transform_list)

    def preprocess_image(self, path):
        img = Image.open(path).convert('RGB')
        return self.transform(img)

class ModelEval:
    def __init__(self, model_path):
        self.opt = TestOptions().parse()
        self.opt.nThreads = 1  # test code only supports nThreads = 1
        self.opt.batchSize = 1  # test code only supports batchSize = 1
        self.opt.serial_batches = True  # no shuffle
        self.opt.no_flip = True  # no flip

        self.model = make_unet_generator(self.opt)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.trans = ProcessImage()

        mother_path = './datasets/lfw/'
        self.pathA = os.path.join(mother_path, 'trainB')
        self.pathB = os.path.join(mother_path, 'trainA')

        self.listA = glob(os.path.join(self.pathA, '*'))

    def view_single(self, idx):
        sample_path = self.listA[idx]
        sample_name = sample_path.split('/')[-1].split('_B.jpg')[0]
        sample_path_Y = os.path.join(self.pathB, sample_name + '_A.jpg')

        inputs = self.trans.preprocess_image(sample_path)
        answer = self.trans.preprocess_image(sample_path_Y)

        y = self.model(inputs.unsqueeze(0))

        plt.imshow(transforms.functional.rotate(inputs, 90).transpose(0, 2))
        plt.show()

        plt.imshow(transforms.functional.rotate(y[0].detach(), 90).transpose(0, 2))
        plt.show()

        plt.imshow(transforms.functional.rotate(answer, 90).transpose(0, 2))
        plt. show()

    def view_list(self, input_list):
        for input in input_list:
            self.view_single(input)


if __name__ == '__main__':
    model_path = './checkpoints/experiment_name/latest_net_G.pth'
    me = ModelEval(model_path)
    print(me.view_list([0]))
    me.view_list([0, 5, 10, 55])
    print(1)
