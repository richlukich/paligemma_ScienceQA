from PIL import Image
from io import BytesIO
import torch

def config_prompt_format(data, qfmt='QO', afmt='A'):

    qfmt_options = {
        'Q': f'{data["question"]}. Options: {data["choices"]}',
        'QO': f'Question: {data["question"]}. Options: {data["choices"]}.',
        'CQO': f'{data["hint"]}. Question: {data["question"]}. Options: {data["choices"]}.',
        'QCO': f'Question: {data["question"]}. Context: {data["hint"]}. Options: {data["choices"]}.'
    }


    afmt_options = {
        'Ia': f'{data["answer_str"]}',
        'A': f'The answer is {data["answer_str"]}.',
        'AS': f'The answer is {data["answer_str"]}. BECAUSE: {data["solution"]}',
        'AL': f'The answer is {data["answer_str"]}. BECAUSE: {data["lecture"]}',
        'ALS': f'The answer is {data["answer_str"]}. BECAUSE: {data["lecture"]} {data["solution"]}'
    }

    if qfmt not in qfmt_options:
        raise ValueError(f"Invalid qfmt value: {qfmt}")
    if afmt not in afmt_options:
        raise ValueError(f"Invalid afmt value: {afmt}")

    return qfmt_options[qfmt], afmt_options[afmt]

def collate_fn(ds, processor, device = 'cuda'):

  prefixes = []
  suffixes = []
  images = []
  qmft = "Q"
  amft = "A"
 
  for data in ds:
    prefix, suffix = config_prompt_format(data,qmft,amft)
    prefixes.append(prefix)
    suffixes.append(suffix)

  imgs = [(Image.open(BytesIO(data['image']['bytes']))).convert("RGB") for data in ds]
  tokens = processor(text=prefixes, images=imgs, suffix=suffixes, return_tensors="pt", padding="longest")
  tokens = tokens.to(torch.bfloat16).to(device)

  return tokens


def processing(ds):

  prefixes = []
  suffixes = []
  images = []
  qmft = "Q"
  amft = "A"

  #for index,data in ds.iterrows():
  for data in ds:
    prefix, suffix = config_prompt_format(data,qmft,amft)
    prefixes.append(prefix)
    suffixes.append(suffix)

  imgs = [(Image.open(BytesIO(data['image']['bytes']))).convert("RGB") for data in ds]


  return prefixes, imgs, suffixes

def clean_string(s):
    return s.strip().replace("\'", '').replace("]", '').replace("[", '').replace(".", '').strip().lower()