import random
import argparse
import numpy as np
from tqdm import tqdm
from logger import get_logger
from tabulate import tabulate
from utils import get_transform

import torch
import torch.nn.functional as F

import GlocalCLIP_lib
from dataset import Dataset
from glocal_prompt_generator import GlocalCLIP_PromptLearner

from visualization import visualizer
from metrics import image_level_metrics, pixel_level_metrics
from scipy.ndimage import gaussian_filter

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset

    logger = get_logger(args.save_path)

    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    results = {}
    metrics = {}
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['imgs_masks'] = []
        results[obj]['anomaly_maps'] = []
        metrics[obj] = {}
        metrics[obj]['pixel-auroc'] = 0
        metrics[obj]['pixel-aupro'] = 0
        metrics[obj]['image-auroc'] = 0
        metrics[obj]['image-ap'] = 0

    GlocalCLIP_parameters = {"Normal_Prompt_Length": args.n_ctx, "Anomaly_Prompt_Length": args.ab_ctx, "Deep_Text_Prompt_Depth": args.depth, "Deep_Text_Prompt_Length": args.t_n_ctx}

    model, _ = GlocalCLIP_lib.load("ViT-L/14@336px", device=device, design_details = GlocalCLIP_parameters)
    model.eval()

    prompt_learner = GlocalCLIP_PromptLearner(model.to("cpu"), GlocalCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path, weights_only=True)
    
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = args.dpam)

    global_prompts, global_tokenized_prompts, local_prompts, local_tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
    prompts = torch.cat([global_prompts, local_prompts], dim = 0)
    tokenized_prompts = torch.cat([global_tokenized_prompts, local_tokenized_prompts], dim = 0)

    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 4), dim = 1)
    text_features = text_features/text_features.norm(dim=-1, keepdim=True)

    global_text_features = text_features[:, :2, :]
    local_text_features = text_features[:, 2:, :]

    model.eval()
    model.to(device)
    similarity_scores = {}
    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        cls_name = items['cls_name']
        cls_id = items['cls_id']
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())

        with torch.no_grad():
            image_features, patch_features = model.encode_image(image, features_list, DPAM_layer = 20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            sim_visual = image_features @ global_text_features.permute(0, 2, 1)
            sim_visual = (sim_visual/0.07).softmax(-1)
            
            sim_visual = sim_visual[:, 0, 1]
            sim_visual = sim_visual.detach().cpu().numpy()
            if cls_name[0] not in similarity_scores:
                similarity_scores[cls_name[0]] = {'normal':[], 'anomaly':[]}
            
            if items['anomaly'] == 0:
                similarity_scores[cls_name[0]]['normal'].append(sim_visual[0])
            else:
                similarity_scores[cls_name[0]]['anomaly'].append(sim_visual[0])

            text_probs = image_features @ global_text_features.permute(0, 2, 1)
            text_probs = (text_probs/0.07).softmax(-1)
            text_probs = text_probs[:, 0, 1]
            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
                    similarity, _ = GlocalCLIP_lib.compute_similarity(patch_feature, local_text_features[0])
                    similarity_map = GlocalCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                    anomaly_map = (similarity_map[...,1] + 1 - similarity_map[...,0])/2.0
                    anomaly_map_list.append(anomaly_map)

            anomaly_map = torch.stack(anomaly_map_list)
            anomaly_map = anomaly_map.sum(dim = 0)

            results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
            anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in anomaly_map.detach().cpu()], dim = 0 )
            results[cls_name[0]]['anomaly_maps'].append(anomaly_map)
            visualizer(items['img_path'], anomaly_map.detach().cpu().numpy(), args.image_size, args.save_path, cls_name)
    
    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    image_f1_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []
    pixel_f1_list = []
    for obj in obj_list:
        table = []
        table.append(obj)
        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
        results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
        if args.metrics == 'image-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            image_f1 = image_level_metrics(results, obj, "image-f1")
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            table.append(str(np.round(image_f1 * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap) 
            image_f1_list.append(image_f1)
        elif args.metrics == 'pixel-level':
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            pixel_f1    = pixel_level_metrics(results, obj, "pixel-f1")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            table.append(str(np.round(pixel_f1 * 100, decimals=1)))
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
            pixel_f1_list.append(pixel_f1)
        elif args.metrics == 'image-pixel-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            image_f1 = image_level_metrics(results, obj, "image-f1")

            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            pixel_f1    = pixel_level_metrics(results, obj, "pixel-f1")

            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            table.append(str(np.round(pixel_f1 * 100, decimals=1)))

            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            table.append(str(np.round(image_f1 * 100, decimals=1)))

            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
            image_f1_list.append(image_f1)
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
            pixel_f1_list.append(pixel_f1)

        table_ls.append(table)

    if args.metrics == 'image-level':
        # logger
        table_ls.append(['mean', 
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_f1_list) * 100, decimals=1))
                        ])
        results = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap', 'image_f1'], tablefmt="pipe")
    elif args.metrics == 'pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_f1_list) * 100, decimals=1))
                        ])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'pixel_f1'], tablefmt="pipe")
    elif args.metrics == 'image-pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_f1_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_f1_list) * 100, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'pixel_f1', 'image_auroc', 'image_ap', 'image_f1'], tablefmt="pipe")
    logger.info("\n%s", results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("GlocalCLIP", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./data/mvtec", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/9_12_4_multiscale_proposed/zero_shot', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints/9_12_4_multiscale_proposed/epoch_15.pth', help='path to checkpoint')
    # model
    parser.add_argument("--dpam", type=int, default=20, help="DPAM layer")
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--ab_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    
    args = parser.parse_args()
    setup_seed(args.seed)
    test(args)
