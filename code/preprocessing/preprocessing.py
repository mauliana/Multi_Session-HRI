from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import pandas as pd
import csv
import json
from tqdm import tqdm
from profanity_check import predict_prob
from transformers import pipeline
import logging
logging.getLogger("transformers").setLevel(logging.WARNING)

def classify_text(text, classifier):
    predicted_label = classifier(text)
    return predicted_label[0]['label']

def clean_str(str):
    str = str.replace(" . ", ". ")
    str = str.replace(" , ", ", ")
    str = str.replace(" ? ", "? ")
    str = str.replace(" ' ", "'")
    str = str.replace(" ’ ", "'")
    return str

def get_emotion(value):
    mapping = {'0': 'other', '1': 'anger', '2': 'disgust', '3': 'fear', '4': 'happiness', '5': 'sad', '6': 'surprise'}
    return mapping.get(str(value), 'Invalid')

def save_as_hf_dataset(path, df, dir_name="hf_rg_dataset"):
    print("Check duplicate data")

    size = len(df)
    df = df.drop_duplicates()

    print(f"    Duplicate found: {size - len(df)}")
    print(f"    Total remining data: {len(df)}")

    # Split the dataset into train and validation
    print("Split data into train and valid set")

    train_df, valid_df = train_test_split(df, test_size=0.20, shuffle=True, random_state=42)

    print(f"    Training size: {len(train_df)}")
    print(f"    Validation size: {len(train_df)}")

    # drop the index column
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    # Format into a huggingface dataset
    print("Start formating datasets into hugging_face format")

    train_data = Dataset.from_pandas(train_df)
    valid_data = Dataset.from_pandas(valid_df)

    dataset = DatasetDict()
    dataset['train'] = train_data
    dataset['valid'] = valid_data

    print(" Done!")
    print(dataset)

    # Save dataset to file
    dataset.save_to_disk(path+dir_name)

    print(f"Dataset is save to {path}{dir_name}")

def process_dd(path, plutchik, classifier):
    # Open file
    text = open(path+'dialogues_text.txt', 'r')
    emo = open(path+'dialogues_emotion.txt', 'r')

    # read data
    data_text = []
    for line in text:
        data_text.append(line)

    data_emo = []
    for line in emo:
        data_emo.append(line)


    print(f"Data length: {len(data_text)}")

    # Initialize an empty list to store the data
    input= []
    target = []
    emotion = []

    # for j in range(len(data_text)):
    print("Processing data is starting...")
    for j in tqdm(range(len(data_text)), desc="Processing data", unit="iteration"):

        split_text = data_text[j].split("__eou__")
        split_emo= data_emo[j].split(" ")

        utt_size = len(split_text)-1

        hist = []
        
        # Filter the data, we only use the multi-turn data
        if utt_size < 3:
            continue
        else:
            for i in range(utt_size-2):
                c_input =clean_str(split_text[i])

                
                if len(hist) == 0: c_hist = ""
                elif len(hist) > 2: c_hist = hist[i-2]+"\n"+hist[i-1]+"\n"
                else: c_hist = hist[i-1]+"\n"

                if i % 2 != 0:
                    text = "S2: "+c_input
                else:
                    text = "S1: "+c_input
                
                hist.append(text)
                input_text = c_hist+text
                target_text = clean_str(split_text[i+1])

                # check for emotion label
                c_emo = get_emotion(split_emo[i])
                if c_emo in plutchik:
                    input_emo = c_emo
                else:
                    input_emo = classify_text(c_input, classifier)
                
                # check profanity for target text, only save for target text with profanity below 80%
                profanity = predict_prob([target_text])
                if profanity > 0.80:
                    continue
                else:
                    input.append(input_text)
                    target.append(target_text)
                    emotion.append(input_emo)

    # Combine the data into a DataFrame
    dd_data = pd.DataFrame({
        'emotion': emotion,
        'input_text': input,
        'target_text': target
    })

    print(" Done!")

    dd_data = remove_duplicate(dd_data)

    return dd_data

def process_ed(path, ed_mapping, classifier):
    # csv_file_path = "../datasets/raw/empatheticdialogues/"

    # Initialize an empty list to store the data
    ed_data = []

    # Open csv file and read its content
    with open(path+"train.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        # Iterate over row in the CSV file
        for row in reader:
            ed_data.append(row)

    with open(path+"valid.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ed_data.append(row)

    # store data into dataframe
    ed_df = pd.DataFrame(ed_data)
    print(f"Dataset size: {len(ed_df)}")

    # To avoid duplicate, get the data based on the conv_id
    conv_id_list = ed_df.conv_id.unique()

    # Initialize an empty list to store the data
    input = []
    target = []
    emotion = []

    print("Start preprocessing...")
    for id in tqdm(conv_id_list, desc="Processing data", unit="iteration"):
    # for id in conv_id_list:
        dialog = ed_df[ed_df['conv_id'] == id]
        size = len(dialog)

        # Check if the current dialog is multi-turn conv
        if size > 2:
            hist = []

            # Process the emotion label
            dialog_emo = dialog['context'].iloc[0]
            # print(dialog_emo)
            
            # Check if the emo part of plutchik
            if dialog_emo in ed_mapping:
                input_emo = ed_mapping[dialog_emo]
                # print(f"mapping: {input_emo}")
            else:
                input_emo = classify_text(dialog_emo, classifier)

            # Process the utterances in each dialog
            for i in range(size-1):
                c_input = dialog['utterance'].iloc[i].replace("_comma_", ",")
                
                # check for history
                if len(hist) == 0: c_hist = ""
                elif len(hist) > 2: c_hist = hist[i-2]+"\n"+hist[i-1]+"\n"
                else: c_hist = hist[i-1]+"\n"


                if i % 2 != 0:
                    text = "S2: "+c_input
                else:
                    text = "S1: "+c_input
                
                hist.append(text)
                input_text = c_hist+text
                target_text = dialog['utterance'].iloc[i+1].replace("_comma_", ",")

                # check profanity for the target
                profanity = predict_prob([target_text])
                if profanity > 0.80:
                    continue
                else:
                    input.append(input_text)
                    target.append(target_text)
                    emotion.append(input_emo)

    # Combine the data into a DataFrame
    ed_data_clean = pd.DataFrame({
        'emotion': emotion,
        'input_text': input,
        'target_text': target
    })

    print("Done!")
    ed_data_clean = remove_duplicate(ed_data_clean)

    return ed_data_clean

def process_tc(json_path, tc_mapping, classifier):
    # json_path = "../datasets/raw/topicalchat/"
    with open(json_path+"train.json", "r") as json_files:
        data = json.load(json_files)
    
    # Flatten the nested structure
    flattened_data = []
    for key, value in data.items():
        for content_item in value['content']:
            flattened_data.append({
                'conversation_id': key,
                'article_url': value['article_url'],
                'config': value['config'],
                'message': content_item['message'],
                'agent': content_item['agent'],
                'sentiment': content_item['sentiment'],
                'knowledge_source': content_item['knowledge_source'],
                'turn_rating': content_item['turn_rating']
            })

    # Create a DataFrame
    df = pd.DataFrame(flattened_data)

    # remove unused attributes
    df = df.drop(columns=['article_url', 'config', 'agent', 'turn_rating'])
    
    conv_id = df.conversation_id.unique()

    # Initialize an empty list to store the data
    input = []
    target = []
    emotion = []

    # for i in range (len(conv_id)):
    print("Start preprocessing...")
    for i in tqdm(conv_id, desc="Processing data", unit="iteration"):
        dialog = df[df['conversation_id'] == i]
        size = len(dialog)

        if size > 2:
            hist = []
            for j in range(size-1):
                # check the emotion label
                dialog_emo = dialog['sentiment'].iloc[j]
                c_input = dialog['message'].iloc[j]

                # Check if the emo part of plutchik
                if dialog_emo in tc_mapping:
                    input_emo = tc_mapping[dialog_emo]
                    # print(f"mapping: {input_emo}")
                else:
                    input_emo = classify_text(dialog_emo, classifier)
                    # print(f"classify: {input_emo}")

                # check for history
                if len(hist) == 0: c_hist = ""
                elif len(hist) > 2: c_hist = hist[j-2]+"\n"+hist[j-1]+"\n"
                else: c_hist = hist[j-1]+"\n"

                if j % 2 != 0:
                    text = "S2: "+c_input
                else:
                    text = "S1: "+c_input
                
                hist.append(text)
                input_text = c_hist+text
                target_text = dialog['message'].iloc[j+1]

                # check profanity for the target
                profanity = predict_prob([target_text])
                if profanity > 0.80:
                    continue
                else:
                    input.append(input_text)
                    target.append(target_text)
                    emotion.append(input_emo)
                # print(f'emotion: {input_emo} \ninput: \n{input_text}\ntarget: {target_text}')

    # Combine the data into a DataFrame
    data_clean = pd.DataFrame({
        'emotion': emotion,
        'input_text': input,
        'target_text': target
    })

    print("Done!")
    data_clean = remove_duplicate(data_clean)

    return data_clean

def prompting(sample):
    # prompt = [f"Consider the {item['emotion']} feeling, predict the next response for: {item['input_text']}" for _, item in sample.iterrows()]
    prompt = [f"Current utterance emotions are a {item['emotion']}. By considering the emotion, predict the next response: {item['input_text']}" for _, item in sample.iterrows()]
    return prompt

def read_csv_file(path):
    data = pd.read_csv(path, sep='\t').astype(str)
    return data

def remove_duplicate(df):
    print(f"Data size: {len(df)} \nStart removing the duplicate")
    unique_df = df.drop_duplicates()
    print(f"    Duplicate found: {len(df) - len(unique_df)}")
    print(f"    Total remining data: {len(unique_df)}")
    return unique_df

def emo_prompting(sample):
    prompt = "Please predict the Plutchik's emotion label for this utterance:"
    text = [prompt + item['text'] for _, item in sample.iterrows()]
    data = pd.DataFrame({
        'input_text': text,
        'target_text': sample['label']
    })
    return data

def pb_prompting(sample):
    prompt = "Please predict if this utterance contains personal background information:"
    text = [prompt + item['text'] for _, item in sample.iterrows()]
    data = pd.DataFrame({
        'input_text': text,
        'target_text': sample['label']
    })
    return data

def process_clf_data(path):
    dd = read_csv_file(path+'dd_plutchik_label.tsv')
    ed = read_csv_file(path+'emo_plutchik_label.tsv')
    bst = read_csv_file(path+'bst_personal_background.tsv')
    tc = read_csv_file(path+'tc_personal_background.tsv')

    # concate dataset
    data_emo = pd.concat([dd, ed])
    data_pb = pd.concat([bst, tc])

    # remove duplicate
    print("\nEmo data:")
    data_emo = remove_duplicate(data_emo)

    print("\nPersonal background data:")
    data_pb = remove_duplicate(data_pb)

    prompt_emo = emo_prompting(data_emo)
    prompt_pb = pb_prompting(data_pb)

    print(f"\nemotion total data: {len(prompt_emo)}")
    print(f"personal background total data: {len(prompt_pb)}")

    # concate dataset
    data_clean = pd.concat([prompt_emo, prompt_pb])
    return data_clean

def main():
    classifier = pipeline("text-classification", model="helper_models/emo_clf/best-model", return_all_scores=False)
    plutchik = ['anticipating', 'joy', 'trust', 'fear', 'surprise', 'sad', 'disgust', 'anger']

    ed_mapping = {
        'anxious': 'anticipating',
        'joyful': 'joy',
        'content': 'joy',
        'trusting': 'trust',
        'afraid': 'fear',
        'terrified': 'fear',
        'surprised': 'surprise',
        'lonely': 'sad',
        'devastated': 'sad',
        'disgusted': 'disgust',
        'angry': 'anger',
        'annoyed': 'anger',
        'furious': 'anger'
    }

    tc_mapping = {
        'Surprised': 'surprise',
        'Disgusted' : 'disgust', 
        'Fearful': 'fear', 
        'Angry': 'anger'
    }

    # Daily Dialogue dataset
    print("Preprocessing Daily Dialogue dataset...")
    dd_path = '../datasets/raw/dailydialog/'
    dd_data = process_dd(dd_path, plutchik, classifier)

    # Empathetic Dialogue dataset
    print("\nPreprocessing Empathetic Dialogue dataset...")
    ed_path = '../datasets/raw/empatheticdialogues/'
    ed_data = process_ed(ed_path, ed_mapping, classifier)

    # Topical-chat dataset
    print("\nPreprocessing Topical-chat dataset...")
    tc_path = '../datasets/raw/topicalchat/'
    tc_data = process_tc(tc_path, tc_mapping, classifier)

    # Clean data for clasification task
    print("\nPreprocessing classification data...")
    clf_path ="../datasets/augmented_data/"
    clf_data = process_clf_data(clf_path)


    # Combine datasets
    combined_data = pd.concat([dd_data, ed_data, tc_data, clf_data])

    prompt = prompting(combined_data)
    data= pd.DataFrame({
        'input_text': prompt,
        'target_text': combined_data['target_text']
    })

    output_path = '../datasets/cleaned/'

    save_as_hf_dataset(output_path, data, dir_name="hf_combined_data_v2")


if __name__ == "__main__":
    main()