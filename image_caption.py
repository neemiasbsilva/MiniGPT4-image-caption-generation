import argparse
import os
import random
import pandas as pd
from tqdm import tqdm
import warnings

import numpy as np
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import gradio as gr

from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", help="path to configuration file.", default="eval_configs/minigpt4_eval.yaml")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument('-p', '--data_path')
    parser.add_argument('-b', '--num_beams')
    parser.add_argument('-t', '--temperature')
    parser.add_argument('-s', '--save_path')
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

# ========================================
#             Model Initialization
# ========================================

conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
            'pretrain_llama2': CONV_VISION_LLama2}

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
CONV_VISION = conv_dict[model_config.model_type]

def load_image(image_path, chat_state=CONV_VISION.copy()):
    img_list = [image_path]
    chat.encode_img(img_list)
    return chat_state, img_list

def gradio_ask(user_message, chat_state=CONV_VISION.copy()):
    chat.ask(user_message, chat_state)
    return '', chat_state


def gradio_answer(chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(
        conv=chat_state, 
        img_list=img_list, 
        num_beams=num_beams, 
        temperature=temperature,
        max_new_tokens=300,
        max_length=2000
    )[0]
    return llm_message

def save_captions(df, save_path):
    df.to_csv(os.path.join(save_path, 'image_caption.csv'), index=False)


def main():
    global args
    warnings.filterwarnings("ignore")

    data_path = args.data_path
    img_names = os.listdir(data_path)
    num_beams = int(args.num_beams)
    temperature = float(args.temperature)
    save_path = args.save_path

    df = pd.DataFrame({})

    for img_name in tqdm(img_names, desc="Image captioning progress"):
        image_path = os.path.join(data_path, img_name)
        chat_state, image_list = load_image(image_path=image_path)
        _, chat_state = gradio_ask("Describe this image bring all the details.")
        llm_message = gradio_answer(chat_state, image_list, num_beams, temperature)
        df = pd.concat([df, pd.DataFrame({"image_path": [str(image_path)], "caption": [str(llm_message)]})], axis=0)

    save_captions(df, save_path)


if __name__ == "__main__":
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [[835], [2277, 29937]]
    stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
    print('Initialization Finished')

    main()