import os
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

import GlocalCLIP_lib
from glocal_prompt_generator import GlocalCLIP_PromptLearner, TripletLoss
from loss import FocalLoss, BinaryDiceLoss

from dataset import Dataset
from logger import get_logger
from utils import get_transform

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):

    logger = get_logger(args.save_path)

    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    GlocalCLIP_parameters = {"Normal_Prompt_Length": args.n_ctx, "Anomaly_Prompt_Length": args.ab_ctx, "Deep_Text_Prompt_Depth": args.depth, "Deep_Text_Prompt_Length": args.t_n_ctx}

    model, _ = GlocalCLIP_lib.load("ViT-L/14@336px", device=device, design_details = GlocalCLIP_parameters)

    prompt_learner = GlocalCLIP_PromptLearner(model.to("cpu"), GlocalCLIP_parameters)
    prompt_learner.to(device)
    
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = args.dpam)

    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(list(prompt_learner.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    # hyperparameters
    alpha = args.alpha 

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    contrastive_loss_fn = TripletLoss(margin=args.margin)

    model.eval()
    model.visual.eval()
    prompt_learner.train()
    for epoch in tqdm(range(args.epoch)):

        loss_list = []
        image_loss_list = []

        for items in tqdm(train_dataloader):
            image = items['img'].to(device)
            label =  items['anomaly']

            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            with torch.no_grad():
                image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer = 20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            global_prompts, global_tokenized_prompts, local_prompts, local_tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
            prompts = torch.cat([global_prompts, local_prompts], dim = 0)
            tokenized_prompts = torch.cat([global_tokenized_prompts, local_tokenized_prompts], dim = 0) 

            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 4), dim = 1)
            text_features = text_features/text_features.norm(dim=-1, keepdim=True)                              # [1, 4, 768]

            global_text_features = text_features[:, :2, :]
            local_text_features = text_features[:, 2:, :]

            text_probs = image_features.unsqueeze(1) @ global_text_features.permute(0, 2, 1)
            text_probs = text_probs[:, 0, ...]/0.07

            image_loss = F.cross_entropy(text_probs.squeeze(), label.long().to(device))
            image_loss_list.append(image_loss.item())

            similarity_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
                    similarity, _ = GlocalCLIP_lib.compute_similarity(patch_feature, local_text_features[0])
                    similarity_map = GlocalCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
                    similarity_map_list.append(similarity_map)

            loss_normal   = contrastive_loss_fn(global_text_features[:, 0, :], local_text_features[:, 0, :], local_text_features[:, 1, :])
            loss_abnormal = contrastive_loss_fn(global_text_features[:, 1, :], local_text_features[:, 1, :], local_text_features[:, 0, :])
            contrastive_loss = loss_normal + loss_abnormal

            loss = 0
            for i in range(len(similarity_map_list)):
                loss += loss_focal(similarity_map_list[i], gt)
                loss += loss_dice(similarity_map_list[i][:, 1, :, :], gt)
                loss += loss_dice(similarity_map_list[i][:, 0, :, :], 1-gt)

            optimizer.zero_grad()
            (loss+image_loss+alpha*contrastive_loss).backward()
            optimizer.step()
            loss_list.append(loss.item())

        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({"prompt_learner": prompt_learner.state_dict()}, ckp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("GlocalCLIP", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoints/9_12_4_multiscale_proposed/', help='path to save results')

    parser.add_argument("--dataset", type=str, default='visa', help="train dataset name")
    parser.add_argument("--dpam", type=int, default=20, help="dpam layer")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--ab_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--margin", type=float, default=0.0, help="margin")
    parser.add_argument("--alpha", type=float, default=1.0, help="alpha")
    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
