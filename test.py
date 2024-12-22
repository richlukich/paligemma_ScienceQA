from transformers import AutoTokenizer, PaliGemmaForConditionalGeneration, PaliGemmaProcessor
import torch
from model_lora import bnb_config
from transformers import AutoProcessor
from datasets import load_dataset
from utils import processing, clean_string
from sklearn.metrics import accuracy_score

def testing(questions,images,answers):
  predicted = []
  with torch.inference_mode():
    for i in range(len(questions)):
      inputs = processor(questions[i],images[i] , return_tensors="pt")
      inputs.to('cuda')
      output = model.generate(**inputs, max_new_tokens=20)
      o = processor.decode(output[0], skip_special_tokens=True)[len(questions[i])+1:]
      print(f'{i} {answers[i]} | {o}')
      predicted.append(o)

  return predicted

if __name__ == '__main__':

    device="cuda"
    model_id = "richlukich/paligemma_scienceQA"
    token = "<your token>"
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                            quantization_config=bnb_config, device_map={"":0})
    model_id = "google/paligemma-3b-pt-224"
    processor = AutoProcessor.from_pretrained(model_id,token=token)

    

    dataset = load_dataset("richlukich/scienceQAv1")
    test_q, test_i, test_a = processing(dataset['test'])
    val_q, val_i, val_a = processing(dataset['validation'])

    predicted_val = testing(val_q, val_i, val_a)

    val_a = [clean_string(s) for s in val_a]
    predicted_val = [clean_string(s) for s in predicted_val]
    accuracy_score(val_a, predicted_val)
    print (accuracy_score(val_a, predicted_val))