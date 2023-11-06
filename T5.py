"""
Author: Hackathon
Version: 1.0 06-11-2023
Description: Tokenize the CSV file of AOP-ALL using T5-small pre-trained model from Hugging Face
Notes: The current version is based on a dataset from AOP help finder and PubMed.
       The path to the files should be modified to user-specified locations for proper execution of the script.
Potential issues: no known issues
"""
from transformers import T5Tokenizer
import pandas as pd
import torch  
tokenizer = T5Tokenizer.from_pretrained("t5-small")
csv_file = 'C:/Users/Losti/Desktop/AOP-ALL.csv'
df = pd.read_csv(csv_file, encoding='latin1')
inputs = df['inputs']
outputs = df['outputs']
tokenized_inputs = tokenizer(inputs.to_list(), return_tensors="pt", padding="max_length", max_length=128, truncation=True)
tokenized_outputs = tokenizer(outputs.to_list(), return_tensors="pt", padding="max_length", max_length=128, truncation=True)
save_path = 'C:/Users/Losti/Desktop/tokenized_data.pt'
torch.save({'input_ids': tokenized_inputs['input_ids'], 'attention_mask': tokenized_inputs['attention_mask'],
            'decoder_input_ids': tokenized_outputs['input_ids'], 'decoder_attention_mask': tokenized_outputs['attention_mask']},
           save_path)
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd
import torch
tokenized_data = torch.load('C:/Users/Losti/Desktop/tokenized_data.pt')
csv_file = 'C:/Users/Losti/Desktop/AOP-ALL.csv'
df = pd.read_csv(csv_file, encoding='latin1')
inputs = df['inputs']
outputs = df['outputs']
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)
tokenized_inputs = tokenizer(inputs.to_list(), return_tensors="pt", padding="max_length", max_length=128, truncation=True)
tokenized_outputs = tokenizer(outputs.to_list(), return_tensors="pt", padding="max_length", max_length=128, truncation=True)
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    evaluation_strategy="steps",
    eval_steps=1000,  # Adjust as needed
    save_steps=1000,  # Adjust as needed
    num_train_epochs=3,
    save_total_limit=2,
    logging_dir="./logs",
    remove_unused_columns=False  # Important: Set this to False to prevent the 'labels' issue
)
dataset = torch.utils.data.TensorDataset(
    tokenized_inputs['input_ids'],
    tokenized_inputs['attention_mask'],
    tokenized_outputs['input_ids'],
    tokenized_outputs['attention_mask']
)
def my_data_collator(features):
    batch = {
        'input_ids': torch.stack([feature[0] for feature in features]),
        'attention_mask': torch.stack([feature[1] for feature in features]),
        'labels': torch.stack([feature[2] for feature in features]),
        'decoder_attention_mask': torch.stack([feature[3] for feature in features]),
    }
    return batch
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=my_data_collator,
    train_dataset=dataset,
)
trainer.train()
validation_csv_file = 'C:/Users/Losti/Desktop/dataset01.csv'  # Adjust the path
validation_df = pd.read_csv(validation_csv_file, encoding='latin1')
val_inputs = validation_df['inputs']
val_outputs = validation_df['outputs']
tokenized_val_inputs = tokenizer(list(val_inputs), return_tensors="pt", padding=True, truncation=True, max_length=128, return_attention_mask=True)
tokenized_val_outputs = tokenizer(list(val_outputs), return_tensors="pt", padding=True, truncation=True, max_length=128, return_attention_mask=True)
validation_dataset = torch.utils.data.TensorDataset(
    tokenized_val_inputs['input_ids'],
    tokenized_val_inputs['attention_mask'],
    tokenized_val_outputs['input_ids'],
    tokenized_val_outputs['attention_mask']
)
results = trainer.predict(validation_dataset)
print(results)

