from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("speedcell4/wmt14-deen-shared-40k")
# tokenizer.save_pretrained("tokenizers/wmt14-deen-shared-40k")
# print(tokenizer.vocab_size)