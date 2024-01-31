import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
#from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
#import habana_frameworks.torch.gpu_migration
import habana_frameworks.torch.core as htcore

import time

#adapt_transformers_to_gaudi()

HPU = torch.device("hpu")

#torch.set_default_device(HPU)

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.bfloat16, trust_remote_code=True)

model = wrap_in_hpu_graph(model. eval().to(HPU))

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", torch_dtype=torch.bfloat16, trust_remote_code=True)

inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

inputs = inputs.to(HPU)

print("BEFORE GENERATE")

start_generation_time = time.time()

outputs = model.generate(**inputs, max_length=200)

print("AFTER GENERATE")

end_generation_time = time.time()

text = tokenizer.batch_decode(outputs)[0]
print(text)

print("Generation Time: {}".format(end_generation_time - start_generation_time))

#############################################################################################

inputs = tokenizer('''def print_odd(n):
   """
   Print all odd between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

inputs = inputs.to(HPU)

print("BEFORE GENERATE")

start_generation_time = time.time()

outputs = model.generate(**inputs, max_length=200)

print("AFTER GENERATE")

end_generation_time = time.time()

text = tokenizer.batch_decode(outputs)[0]
print(text)

print("Generation Time: {}".format(end_generation_time - start_generation_time))

