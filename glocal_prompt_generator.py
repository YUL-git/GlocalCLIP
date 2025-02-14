import torch
import torch.nn as nn
import torch.functional as F
from typing import Union, List
from pkg_resources import packaging
from GlocalCLIP_lib.simple_tokenizer import SimpleTokenizer as _Tokenizer
from copy import deepcopy

_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

def encode_text_with_prompt_ensemble(model, texts, device):
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.', 'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(texts[0]) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence)
        class_embeddings = model.encode_text(prompted_sentence.to(device))
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        text_features.append(class_embedding)

    text_features = torch.stack(text_features, dim=1).to(device).t()

    return text_features

def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])

class GlocalCLIP_PromptLearner(nn.Module):
    def __init__(self, clip_model, design_details):
        super().__init__()
        classnames = ["object"]
        self.n_cls = len(classnames)
        self.n_ctx = design_details["Normal_Prompt_Length"]
        self.ab_ctx = design_details["Anomaly_Prompt_Length"]
        n_ctx_pos = self.n_ctx
        n_ctx_neg = self.ab_ctx
        self.text_encoder_n_ctx = design_details["Deep_Text_Prompt_Length"] 
        dtype = clip_model.transformer.get_cast_dtype()

        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        self.classnames = classnames

        self.state_normal_list = [
            "{}",
        ]

        self.state_anomaly_list = [
            'damaged {}',
        ]
        
        normal_num = len(self.state_normal_list)
        anormaly_num = len(self.state_anomaly_list)
        self.normal_num = normal_num
        self.anormaly_num = anormaly_num

        self.ctx_global_pos = nn.Parameter(torch.empty(self.n_cls, 1, n_ctx_pos, ctx_dim, dtype=dtype))
        self.ctx_global_neg = nn.Parameter(torch.empty(self.n_cls, 1, n_ctx_neg, ctx_dim, dtype=dtype))
        self.ctx_local_pos = nn.Parameter(torch.empty(self.n_cls, 1, n_ctx_pos, ctx_dim, dtype=dtype))
        self.ctx_local_neg = nn.Parameter(torch.empty(self.n_cls, 1, n_ctx_neg, ctx_dim, dtype=dtype))

        nn.init.normal_(self.ctx_global_pos, std=0.02)
        nn.init.normal_(self.ctx_global_neg, std=0.02)
        nn.init.normal_(self.ctx_local_pos, std=0.02)
        nn.init.normal_(self.ctx_local_neg, std=0.02)
        
        prompt_prefix_global_pos = " ".join(["G"] * n_ctx_pos)  # G for global
        prompt_prefix_global_neg = " ".join(["G"] * n_ctx_neg)  # G for global

        prompt_prefix_local_pos = " ".join(["L"] * n_ctx_pos)   # L for local
        prompt_prefix_local_neg = " ".join(["L"] * n_ctx_neg)   # L for local

        #---------------------------------------------------------#
        self.compound_prompts_depth = design_details["Deep_Text_Prompt_Depth"]
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                                                        for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)

        single_layer = nn.Linear(ctx_dim, 896)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        #---------------------------------------------------------#

        prompts_global_pos = [prompt_prefix_global_pos + " " + template.format(name)+ "." for template in self.state_normal_list for name in classnames]
        prompts_global_neg = [prompt_prefix_global_pos + " " + prompt_prefix_global_neg +  " " + template.format(name)+ "." for template in self.state_anomaly_list for name in classnames]

        prompt_local_pos = [prompt_prefix_local_pos + " " + template.format(name)+ "." for template in self.state_normal_list for name in classnames]
        prompt_local_neg = [prompt_prefix_local_pos + " " + prompt_prefix_local_neg + " " + template.format(name)+ "." for template in self.state_anomaly_list for name in classnames]

        tokenized_global_prompts_pos = []
        tokenized_global_prompts_neg = []
        tokenized_local_prompts_pos = []
        tokenized_local_prompts_neg = []

        for p_pos in prompts_global_pos:
            tokenized_global_prompts_pos.append(tokenize(p_pos))
        for p_neg in prompts_global_neg:
            tokenized_global_prompts_neg.append(tokenize(p_neg))
        for p_pos in prompt_local_pos:
            tokenized_local_prompts_pos.append(tokenize(p_pos))
        for p_neg in prompt_local_neg:
            tokenized_local_prompts_neg.append(tokenize(p_neg))

        tokenized_prompts_global_pos = torch.cat(tokenized_global_prompts_pos)
        tokenized_prompts_global_neg = torch.cat(tokenized_global_prompts_neg)
        tokenized_prompts_local_pos  = torch.cat(tokenized_local_prompts_pos)
        tokenized_prompts_local_neg  = torch.cat(tokenized_local_prompts_neg)
        
        with torch.no_grad():
            embedding_global_pos = clip_model.token_embedding(tokenized_prompts_global_pos).type(dtype)
            embedding_global_neg = clip_model.token_embedding(tokenized_prompts_global_neg).type(dtype)
            embedding_local_pos = clip_model.token_embedding(tokenized_prompts_local_pos).type(dtype)
            embedding_local_neg = clip_model.token_embedding(tokenized_prompts_local_neg).type(dtype)

            n, l, d = embedding_global_pos.shape
            embedding_global_pos = embedding_global_pos.reshape(normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_global_neg = embedding_global_neg.reshape(anormaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_local_pos = embedding_local_pos.reshape(normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_local_neg = embedding_local_neg.reshape(anormaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)


        self.register_buffer("token_prefix_global_pos", embedding_global_pos[:, :, :1, :] )
        self.register_buffer("token_suffix_global_pos", embedding_global_pos[:, :,1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_global_neg", embedding_global_neg[:,:, :1, :])
        self.register_buffer("token_suffix_global_neg", embedding_global_neg[:, :, 1 + n_ctx_pos + n_ctx_neg:, :])

        self.register_buffer("token_prefix_local_pos", embedding_local_pos[:, :, :1, :] )
        self.register_buffer("token_suffix_local_pos", embedding_local_pos[:, :,1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_local_neg", embedding_local_neg[:,:, :1, :])
        self.register_buffer("token_suffix_local_neg", embedding_local_neg[:, :, 1 + n_ctx_pos + n_ctx_neg:, :])

        n, d = tokenized_prompts_global_pos.shape
        tokenized_prompts_global_pos = tokenized_prompts_global_pos.reshape(normal_num, self.n_cls, d).permute(1, 0, 2)
        n, d = tokenized_prompts_global_neg.shape
        tokenized_prompts_global_neg = tokenized_prompts_global_neg.reshape(anormaly_num, self.n_cls, d).permute(1, 0, 2)
        n, d = tokenized_prompts_local_pos.shape
        tokenized_prompts_local_pos = tokenized_prompts_local_pos.reshape(normal_num, self.n_cls, d).permute(1, 0, 2)
        n, d = tokenized_prompts_local_neg.shape
        tokenized_prompts_local_neg = tokenized_prompts_local_neg.reshape(anormaly_num, self.n_cls, d).permute(1, 0, 2)

        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg

        # tokenized_prompts = torch.cat([tokenized_prompts_pos, tokenized_prompts_neg], dim=0)  # torch.Tensor
        self.register_buffer("tokenized_prompts_global_pos", tokenized_prompts_global_pos)
        self.register_buffer("tokenized_prompts_global_neg", tokenized_prompts_global_neg)
        self.register_buffer("tokenized_prompts_local_pos", tokenized_prompts_local_pos)
        self.register_buffer("tokenized_prompts_local_neg", tokenized_prompts_local_neg)

    def forward(self, cls_id =None):
        
        ctx_global_pos = self.ctx_global_pos
        ctx_global_neg = self.ctx_global_neg
        ctx_local_pos  = self.ctx_local_pos
        ctx_local_neg  = self.ctx_local_neg

        prefix_global_pos = self.token_prefix_global_pos
        prefix_global_neg = self.token_prefix_global_neg
        suffix_global_pos = self.token_suffix_global_pos
        suffix_global_neg = self.token_suffix_global_neg

        prefix_local_pos = self.token_prefix_local_pos
        prefix_local_neg = self.token_prefix_local_neg
        suffix_local_pos = self.token_suffix_local_pos
        suffix_local_neg = self.token_suffix_local_neg

        prompts_global_pos = torch.cat(
            [
                prefix_global_pos,  # (n_cls, 1, dim)
                ctx_global_pos,     # (n_cls, n_ctx, dim)
                suffix_global_pos,  # (n_cls, *, dim)
            ],
            dim=2,
        )

        prompts_global_neg = torch.cat(
            [
                prefix_global_neg,  # (n_cls, 1, dim)
                ctx_global_pos,
                ctx_global_neg,     # (n_cls, n_ctx, dim)
                suffix_global_neg,  # (n_cls, *, dim)
            ],
            dim=2,
        )

        prompts_local_pos = torch.cat(
            [
                prefix_local_pos,   # (n_cls, 1, dim)
                ctx_local_pos,      # (n_cls, n_ctx, dim)
                suffix_local_pos,   # (n_cls, *, dim)
            ],
            dim=2,
        )

        prompts_local_neg = torch.cat(
            [
                prefix_local_neg,   # (n_cls, 1, dim)
                ctx_local_pos,
                ctx_local_neg,      # (n_cls, n_ctx, dim)
                suffix_local_neg,   # (n_cls, *, dim)
            ],
            dim=2,
        )

        _, _, l, d = prompts_global_pos.shape
        prompts_global_pos = prompts_global_pos.reshape(-1, l, d)
        _, _, l, d = prompts_global_neg.shape
        prompts_global_neg = prompts_global_neg.reshape(-1, l, d)
        global_prompts = torch.cat([prompts_global_pos, prompts_global_neg], dim=0)

        _, l, d = self.tokenized_prompts_global_pos.shape
        tokenized_prompts_global_pos = self.tokenized_prompts_global_pos.reshape(-1,  d)
        _, l, d = self.tokenized_prompts_global_neg.shape
        tokenized_prompts_global_neg = self.tokenized_prompts_global_neg.reshape(-1,  d)
        global_tokenized_prompts = torch.cat((tokenized_prompts_global_pos, tokenized_prompts_global_neg), dim = 0)

        _, _, l, d = prompts_local_pos.shape
        prompts_local_pos = prompts_local_pos.reshape(-1, l, d)
        _, _, l, d = prompts_local_neg.shape
        prompts_local_neg = prompts_local_neg.reshape(-1, l, d)
        local_prompts = torch.cat([prompts_local_pos, prompts_local_neg], dim=0)

        _, l, d = self.tokenized_prompts_local_pos.shape
        tokenized_prompts_local_pos = self.tokenized_prompts_local_pos.reshape(-1,  d)
        _, l, d = self.tokenized_prompts_local_neg.shape
        tokenized_prompts_local_neg = self.tokenized_prompts_local_neg.reshape(-1,  d)
        local_tokenized_prompts = torch.cat((tokenized_prompts_local_pos, tokenized_prompts_local_neg), dim = 0)

        return global_prompts, global_tokenized_prompts, local_prompts, local_tokenized_prompts, self.compound_prompts_text

class TripletLoss(nn.Module):
    def __init__(self, margin=0.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.pairwise_distance(anchor, positive, keepdim=True)
        neg_dist = torch.pairwise_distance(anchor, negative, keepdim=True)
        
        loss = torch.mean(pos_dist ** 2 + torch.clamp(self.margin - neg_dist, min=0.0) ** 2)
        return loss

if __name__ == '__main__':
    import GlocalCLIP_lib

    AnomalyCLIP_parameters = {"Abnormal_Prompt_length": 12, "Prompt_length": 12, "learnabel_text_embedding_depth": 9, "learnabel_text_embedding_length": 4}
    model, _ = GlocalCLIP_lib.load("ViT-L/14@336px", device='cuda', design_details = AnomalyCLIP_parameters)
    prompt_learner = GlocalCLIP_PromptLearner(model.to('cpu'), AnomalyCLIP_parameters)
    
    global_prompts, global_tokenized_prompts, local_prompts, local_tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)

    print(global_prompts.shape, global_tokenized_prompts.shape, local_prompts.shape, local_tokenized_prompts.shape)
    # print(prompts, tokenized_prompts, compound_prompts_text)