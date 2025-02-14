import torchvision.transforms as transforms
from torchvision.transforms import RandAugment
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from GlocalCLIP_lib.transform import image_transform
from GlocalCLIP_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD



def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def get_transform(args):
    preprocess = image_transform(args.image_size, is_train=False, mean = OPENAI_DATASET_MEAN, std = OPENAI_DATASET_STD)
    target_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor()
    ])
    preprocess.transforms[0] = transforms.Resize(size=(args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BICUBIC,
                                                    max_size=None, antialias=None)
    preprocess.transforms[1] = transforms.CenterCrop(size=(args.image_size, args.image_size))
    return preprocess, target_transform

def get_aug_transform(args):
    preprocess = image_transform(args.image_size, is_train=False, mean = OPENAI_DATASET_MEAN, std = OPENAI_DATASET_STD)
    target_transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    RandAugment(num_ops=2, magnitude=9),  # 2개 증강을 랜덤 적용, 강도=9
    transforms.ToTensor()
])
    preprocess.transforms[0] = transforms.Resize(size=(args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BICUBIC,
                                                    max_size=None, antialias=None)
    preprocess.transforms[1] = RandAugment(num_ops=2, magnitude=9)
    return preprocess, target_transform