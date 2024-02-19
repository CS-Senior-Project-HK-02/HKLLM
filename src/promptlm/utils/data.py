import pandas as pd
import json


def prepare_dataset_for_inference(df, text_col, class_col, sample_size,supp_columns = None):
    sampled_df = df.sample(n=sample_size, replace=False)
    if supp_columns:
        sampled_df['combined_text'] = sampled_df[text_col].astype(str)
        for col in supp_columns:
            sampled_df['combined_text'] += " " + sampled_df[col].astype(str)
        x_data = sampled_df['combined_text'].tolist()
    else:
        x_data = sampled_df[text_col].tolist()
        
    data_for_inference = {
        'x': x_data,
        'y': sampled_df[class_col].tolist()
    }
    return data_for_inference
    

def generate_shot_examples(data_dict,   shot_examples):
    
    df = pd.DataFrame(data_dict)
    
    shot_examples_dict = {'x': [], 'y':[]}
    
    indicies_to_remove = []
    
    unique_classes = df['y'].unique()
    
    for cls in unique_classes:
        class_samples = df[df['y'] == cls]
        
        n_samples = min(len(class_samples),shot_examples)
        sampled = class_samples.sample(n_samples)
        
        shot_examples_dict['x'].extend(sampled['x'].tolist())
        shot_examples_dict['y'].extend(sampled['y'].tolist())
        
        indicies_to_remove.extend(sampled.index.tolist())
    
    df_remaining = df.drop(indicies_to_remove)
    
    remaining_data_dict = df_remaining.to_dict('list')
    
    return shot_examples_dict,remaining_data_dict
        
        
    
    
def csv_to_json(filepath,filename="data",skip_index=False):
    df = pd.read_csv(filepath)
    if skip_index:
        df = df.iloc[1:]
    df.reset_index(drop=True,inplace=True)
    #adjust this as needed for now
    json_data = [
    {"prompt": row[1], "completion": row[2]}
    for index, row in df.iterrows()
]
   
    with open(filename+'.json','w') as f:
        json.dump(json_data, f)
            
            
def json_to_generative_dataset():
    pass
    
    

def prepare_dataset_for_training():
    pass


def strip_output_column_by_index(df, input_col_index, output_col_index):
    def strip_repeated_input(input_text, output_text):
        # Find the start of the repeated prompt in the completion
        start_index = output_text.find(input_text)
        # If the prompt is found in the completion, remove it
        if start_index != -1:
            # Remove the prompt and return the completion starting after the prompt
            return output_text[:start_index] + output_text[start_index + len(input_text):]
        return output_text
    
    # Use iloc to access the DataFrame by column indices
    for i in range(len(df)):
        input_text = df.iloc[i, input_col_index]
        output_text = df.iloc[i, output_col_index]
        df.iloc[i, output_col_index] = strip_repeated_input(input_text, output_text)
    return df