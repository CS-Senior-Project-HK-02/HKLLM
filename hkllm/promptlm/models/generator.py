import sys
import os

sys.path.append(r"/home/wstigall/pain/")
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
import pandas as pd
from HKLLM.hkllm.promptlm.utils.data import prepare_dataset_for_generator
from HKLLM.hkllm.promptlm.utils.metrics import sample_recall, sample_accuracy, sample_f1_score, sample_precision
from HKLLM.hkllm.promptlm.utils.parsers import multc_parser, parse_output_for_answer
from peft import PeftModel, PeftConfig, AutoPeftModel, AutoPeftModelForCausalLM
from HKLLM.secrets import import_knowledge_file

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

data = r"//home/wstigall/pain/modified_deid_200k_main.csv"
df = pd.read_csv(data)



dataset = prepare_dataset_for_generator(df=df,text_col="narrative1",class_col=None,sample_size=1000,supp_columns=None)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype = torch.bfloat16
    )
model.load_adapter("/home/wstigall/pain/mistral-pn-hkllm-ksu-internal-v0.1f")

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer = tokenizer, 
    device_map="auto",
    torch_dtype=torch.bfloat16
    
)

system_prompt = """[INST]You are the police department's Ace virtual assistant, you are going to read the following narratives
        and return whether they are related to behavioral health, All samples only have one answer.  The classification of the sample
        is based on the current report, references to past events inside of the report do not affect the classification of the report.
        For your response you must always use <Tag> [Answer] </Tag>.
        you will tag these as either Domestic Social, NonDomestic Social, Mental Health, Substance Abuse or Other there are no other tags.
        You are a virtual assistant so do not refer to yourself in first person when giving reasoning. If you find no evidence of a tag the tag is Other
        The text you must classify is as follows: [/INST]"""
        
texts = dataset["x"]
labels = dataset["y"]


full_prompts = [system_prompt+text+"[INST]  Classify the text, you must ONLY use <Tag> [Answer] </Tag> and can choose ONLY one answer, give succint intermediate reasoning[/INST]" for text in texts ]\

extracted_answers = []
for prompt in full_prompts:
    
    
    sequences = pipe(
            prompt,
            do_sample=True,
            max_new_tokens = 300,
            temperature = 0.3,
            top_k=100,
            top_p = 1.0,
            num_return_sequences = 2                                                                                    ,
            repetition_penalty = 1.15
        )

    model_output_one = sequences[0]['generated_text']
    model_output_two = sequences[1]['generated_text']
    keywords = ["Domestic Social","Domestic_Social","Mental Health","Mental_Health","Substance Abuse","Substance_Abuse","NonDomestic_Social","NonDomestic Social","Other","SubstanceAbuse"]
    extracted_answer = (parse_output_for_answer(model_output_one,keywords=keywords,single_output=True))
    extracted_answer_two = (parse_output_for_answer(model_output_two,keywords=keywords,single_output=True))
    print(sequences[0]['generated_text'])
    print(extracted_answer)
    print(sequences[1]['generated_text'])
    print(extracted_answer_two)
    if extracted_answer == None or extracted_answer == []:
        extracted_answer = "Non_Answer"
        extracted_answers.append(extracted_answer)
    else:
        processed_answer = extracted_answer[0].replace(" ","_")
        extracted_answers.append(processed_answer)
    print(len(extracted_answers),len(labels))
accuracy = sample_accuracy(y_true=labels,y_pred=extracted_answers)
precision = sample_precision(y_true=labels,y_pred=extracted_answers,macro=True)
recall = sample_recall(y_true=labels,y_pred=extracted_answers,macro=True)
f1_score = sample_f1_score(y_true=labels,y_pred=extracted_answers,macro=True)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"f1: {f1_score:.2f}")