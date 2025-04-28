import time
import nltk
from datetime import datetime
import os
import pickle
import torch
import textwrap
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from langchain.prompts import PromptTemplate 
from transformers import pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import json
import torch
torch.cuda.empty_cache()

def load_model(path):
    config = PeftConfig.from_pretrained(path)

    # quantize model from model hub
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # load base LLM model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path,  quantization_config=bnb_config,  device_map={"":0})
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    # Load the Lora model
    model = PeftModel.from_pretrained(model, path, device_map={"":0})
    # model.eval()

    return model, tokenizer

def llm_prepared(model, tokenizer):
    pipe = pipeline(
        "text2text-generation",
        model= model,
        tokenizer=tokenizer,
        num_beams=5, 
        max_new_tokens=100,
        # temperature=0,
        top_p = 0.95,
        top_k = 150,
        repetition_penalty=1.15,
        early_stopping=True,
        do_sample=True
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def load_embedding_model(path):
    # embedding
    model_name = "BAAI/bge-base-en"
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

    embedding = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cuda'},
        encode_kwargs=encode_kwargs
    )

    # Load from disk
    vectordb = Chroma(persist_directory=path, embedding_function=embedding)
    return vectordb

def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')

    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

DEFAULT_SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    if a question refers to you, be mindful that you are a humanoid robots named Emah.
    If a question does not make any sense, or is not factually coherent, please say don't know and don't share false information.
    If the question about the person you talking to, refers to the relevant information below:

    Relevant Information: 
     """

    
def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT, relevant_info=''):
    prompt_template = new_system_prompt + relevant_info + instruction
    return prompt_template

def retrieve_local_info(vectordb, llm, text):
    instruction = """CONTEXT:/n/n {context}/n
    
    Question: {question}"""
    prompt_template = get_prompt(instruction)
    qa_prompt =  PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Retrieval local info
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=retriever,
                                    chain_type_kwargs={"prompt": qa_prompt},
                                    return_source_documents=True)
    llm_response = qa_chain(text)
    result = llm_response['result']

    # return process_llm_response(llm_response)
    return(wrap_text(result))

def generate_response_chain(llm, text, history):
    emo_prompt = PromptTemplate(
        input_variables=["text_input"],
        template="Please predict the Plutchik's emotion label for this utterance: \n {text_input}"
    )
    emo_chain = LLMChain(llm=llm, prompt=emo_prompt)
    emotion = emo_chain.run(text)

    instruction = """Current utterance emotions are a {emotion}. By considering the emotion, predict the next response: 
            {history}"""
    prompt_template = get_prompt(instruction)
    res_prompt =PromptTemplate(template=prompt_template, input_variables=["emotion", "history"])
    response_chain = LLMChain(llm=llm, prompt=res_prompt)
    history = format_dialogue(history)
    temp = {
        "emotion": emotion,
        "history" : history}
    response = response_chain.run(temp)

    return response, emotion

def clf_inference(model, tokenizer, prompt):
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = model.generate(
        input_ids=input_ids, 
        max_new_tokens=10,
        do_sample=True,
        top_k = 150,
        top_p=0.9
        )
    label = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
    return label

def gen_text_inference(model, tokenizer, prompt, p_value=0.95):
    # print(prompt)
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = model.generate(
        input_ids=input_ids, 
        num_beams=5,
        max_new_tokens=100,
        early_stopping=True,
        top_k = 150,
        do_sample=True, 
        repetition_penalty=1.15,
        top_p=p_value
        )
    
    text = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
    return text

def get_bg_info(model, tokenizer, text):
    prompt_template = """Please predict if this utterance contains personal background information:  {text}"""
    prompt = prompt_template.format(text=text)
    bg_info = clf_inference(model, tokenizer, prompt)

    return bg_info

def get_emotion(model, tokenizer, text):
    emo_prompt_template = """Please predict the Plutchik's emotion label for this utterance: {text}"""
    emo_prompt = emo_prompt_template.format(text=text)

    emotion = clf_inference(model, tokenizer, emo_prompt)
    return emotion

def generate_response(model, tokenizer, text, history):
    # get emotion
    emotion = get_emotion(model, tokenizer, text)

    # generate response
    history = format_dialogue(history)
    instruction = """
Current utterance emotions are a {emotion}. By considering the emotion, predict the next response: 
{history}"""
    res_prompt_template = get_prompt(instruction)
    response_prompt = res_prompt_template.format(emotion=emotion, history=history)
    
    response = gen_text_inference(model, tokenizer, response_prompt)
    return response, emotion

def is_question(vectorizer, clf, text):
    question_types = ["whQuestion","ynQuestion"]
    text_type = clf.predict(vectorizer.transform([text]))
    if text_type in question_types:
        return True
    else: 
        return False

def wrap_text(text, width=110):  # used in retrieval_local_info func.
    lines = text.split('\n')
    
    # wrap each line individually
    wrapped_line = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrappedlines back
    wrapped_text = '\n'.join(wrapped_line)

    return wrapped_text

def format_dialogue(utterances):   # used in generate_response func.
    # Determine the number of utterances to format (up to the last 4)
    num_utterances_to_format = min(len(utterances), 2)

    # Format the last 'num_utterances_to_format' utterances
    formatted_utterances = ""
    for i in range(-num_utterances_to_format, 0):
        formatted_utterances += f"{'S1' if i % 2 == 0 else 'S2'}: {utterances[i]}\n"

    return formatted_utterances

def format_dialogue_history(conversation):   # used for history
    formatted_conversation = []

    # Iterate through the conversation array
    for i, message in enumerate(conversation):
        # Determine if the message is from the robot or the human
        speaker = 'Robot' if i % 2 == 0 else 'Human'
        
        # Format the message
        formatted_message = f"{speaker}: {message}"
        
        # Append the formatted message to the new array
        formatted_conversation.append(formatted_message)

    return formatted_conversation

def is_similar(sentence1, sentence2):
    vectorizer = CountVectorizer()

    vectors = vectorizer.fit_transform([sentence1, sentence2]).toarray()
    score = cosine_similarity([vectors[0]], [vectors[1]])
    return score[0][0]

def check_repetition(text):
    # check if the generated sentence contain repetition sentences.
    # print("Check sentence repetition")
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    if len(sentences) == 1:
        return sentences[0]

    # Create a set to store unique sentences
    unique_sentences = set()

    # Create a list to store sentences with only unique sentences
    sentences_with_unique = []

    # Iterate over each sentence
    for sentence in sentences:
        # If the sentence is unique, add it to both the set and the list
        if sentence not in unique_sentences:
            unique_sentences.add(sentence)
            sentences_with_unique.append(sentence)

    if len(sentences_with_unique) > 1:
        text = " ".join(sentences_with_unique)
    else:
        text = sentences_with_unique[0]

    return text

def post_processing(text, history, lm, lm_tokenizer):
    # check if the current generated text similar to the previous one
    # print("\n---POST-PROCESSING STEPS---")
    # print("Human text: "+history[-1])
    # print("current generatede text: "+text)
    similarity = is_similar(text, history[-2])
    # print(f"similarity score: {similarity}")
    if similarity > 0.6:
        # print("same content more than 60% from previous generated text: "+history[-2])
        # generate another response by giving only the user input
        inst = "Predict the next response: "+history[-1]
        prompt = get_prompt(inst)
        new_response = gen_text_inference(lm, lm_tokenizer, prompt)
        # print("new text: "+new_response)

        # recheck the generated result similarity
        if is_similar(new_response, text) > 0.6:
            # if still generate the same result, generate another one with difference top_p value
            new_response = gen_text_inference(lm, lm_tokenizer, prompt, p_value=0.80)
            # print("with dif. p: "+new_response)

        clean_text = check_repetition(new_response)
        
    else:
        clean_text = check_repetition(text)
    return clean_text

def detect_names(sentence, nlp):
    # Process the input sentence with SpaCy
    doc = nlp(sentence)
    words = sentence.split()

    # Extract named entities (persons) from the processed sentence
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    if not names:
        if not words:
            name = sentence
        else:
            name = words[0]
    elif len(names) > 1:
        name = '_'.join(names)
    else: 
        name = names[0]
    # print(name)
    return name

def create_user_profile(path, username, conv_history, recorded_time, summary, personal_background):
    # Create a list of dictionaries for utterance and time pairs
    conversation_data = []

    for utterance, time in zip(conv_history, recorded_time):
        pair = {
            "utterance": utterance,
            "time": time
        }
        conversation_data.append(pair)

    # Create a dictionary to represent the overall data
    data = {
        "conversation_data": conversation_data,
        "personal_background_data": personal_background,
        "summary": summary
    }

    # Convert the dictionary to JSON format
    json_data = json.dumps(data, indent=2)  # indent for better readability

    # Specify the file name
    file_name = f"{path}/{username}.json"

    # Write the JSON data to the file
    with open(file_name, 'w') as json_file:
        json_file.write(json_data)

    print(f"Data has been successfully saved to {file_name}")

def add_data_to_user_profile(file_name, existing_conversation_data, new_conv_history, new_recorded_time, summary, personal_background):
    # Append new data to the existing conversation data
    for utterance, time in zip(new_conv_history, new_recorded_time):
        pair = {
            "utterance": utterance,
            "time": time
        }
        existing_conversation_data.append(pair)
    
    # Update the overall data dictionary
    updated_data = {
        "conversation_data": existing_conversation_data,
        "personal_background_data": personal_background,
        "summary": summary
    }

    # Convert the updated dictionary to JSON format
    json_data = json.dumps(updated_data, indent=2)  # indent for better readability

    # Write the updated JSON data back to the file
    with open(file_name, 'w') as json_file:
        json_file.write(json_data)

    print(f"New data has been successfully appended to {file_name}")


def main():
    start_time = time.time()
    print('start setup...')
    model_path = 'models/lm'
    embedding_path = 'models/db-bge-py'
    user_data_path = 'user_data'
    summarizer_model_path = 'models/summerizer'
    new_profile = ''
    conv_history = []
    recorded_time= []
    pbg_info_list = []

    # Load fine-tuned model
    model, tokenizer = load_model(model_path)

    # Setup llm
    llm = llm_prepared(model, tokenizer)

    # Load embedding
    db = load_embedding_model(embedding_path)

    # additional ML classifier for is_question func
    with open('models/gb_model.pklgb_model.pkl', 'rb') as fin:
        vectorizer, gb_clf = pickle.load(fin)
    
    # Load the SpaCy English model for detect_names func.
    nlp = spacy.load("en_core_web_sm") #if this line throw error run this in terminal: python3 -m spacy download en_core_web_sm

    # Identify the user
    print("Robot: I would like to identify you, could you tell me your name?")

    # Get non-empty input from the user
    while True:
        user_ans = input("You: ")

        # Check if the input is not empty
        if user_ans.strip():
            break  # Exit the loop if the input is non-empty
        else:
            print("Robot: I am sorry, I don't get your answer, could you repeat it again?")

    # Detect the name
    user_name = detect_names(user_ans, nlp)
    print(f"Robot: Thank you {user_name}, Please wait a moment, I am tracing my memory")

    # Check user profile
    files_in_directory = os.listdir(user_data_path)
    profile_file = f"{user_name}.json"
    if profile_file in files_in_directory:
        first_robot_text = f"I remember you, welcome back {user_name}"

        # Read existing data from the file
        with open(user_data_path+'/'+profile_file, 'r') as json_file:
            existing_data = json.load(json_file)

        # Extract existing conversation data and get the summary
        existing_conversation_data = existing_data.get("conversation_data", [])
        existing_personal_background_data = existing_data.get("personal_background_data", [])
        summary = existing_data.get("summary", "")
        print(f"Summary from previous conv: {summary}")
    else:
        first_robot_text = "I think we have never met before, I will remember you for the next time. Let us get started. How are you doing today?"
        
        new_profile = user_name

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Setup done in: {elapsed_time} seconds\n")

    print("-----Start the chatbot----")
    
    conv_history.append(first_robot_text)
    robot_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    recorded_time.append(robot_time)   

    print("type 'bye' to quit the conversation")
    print("Robot: "+conv_history[0])


    while True:
        user_input = input("You: ")
        if user_input == "bye":
            break
        
        # Get the time
        user_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        
        # Store user response to conv_history
        conv_history.append(user_input)
        recorded_time.append(user_time)

        # Detect whether the human response contain background information
        bg_info = get_bg_info(model, tokenizer, user_input)
        if bg_info == "Yes":
            pbg_info_list.append(user_input)
        
        print("--------------------------------------")
        
        if(is_question(vectorizer, gb_clf, user_input)):
            print("___RAG route___")
            response = retrieve_local_info(db, llm, user_input)
            
        else:
            print("___LLM route___")
            # response, emotion = generate_response_chain(llm, user_input, conv_history)
            response, emotion = generate_response(model, tokenizer, user_input, conv_history)
            print("The Human emotion detected: "+emotion)
        
        # post_text = post_processing(response, conv_history, similarity_model, model, tokenizer)
        post_text = post_processing(response, conv_history, model, tokenizer)
        robot_emotion = get_emotion(model, tokenizer, response)
        
        print("The Robot emotion detected: "+robot_emotion)
        print("Background info detection: "+bg_info)
        print("--------------------------------------\n")
        # print("Robot: "+response)
        print("Robot: "+post_text)
        robot_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        recorded_time.append(robot_time) 
        conv_history.append(post_text)
    
    print("\nHistory of this conversation session:")
    hist_data = format_dialogue_history(conv_history)
    hist_data_text = '\n'.join(hist_data)
    print(hist_data_text)
    # for x in conv_history:
    #     print(x)

    if len(pbg_info_list) != 0:
        print("\nDetected personal background information:")
        pbi_text = '\n'.join(pbg_info_list)
        for y in pbg_info_list:
            print(y)   
    else:
        pbi_text = ''
    
    # Load summarizer model
    sum_model, sum_token = load_model(summarizer_model_path)

    # write user profile info into file
    if not new_profile:  #if new profile variable is not empty
        # get conversation summary
        prev_conv_text = []
        for item in existing_conversation_data:
            prev_conv_text.append(item['utterance'])

        conv_data = prev_conv_text + hist_data
        conv_data = '\n'.join(conv_data)
        summary = gen_text_inference(sum_model, sum_token, conv_data)

        # personal background data
        pbi_data = existing_personal_background_data + pbg_info_list
        pbi_data = '\n'.join(pbi_data)

        # append new data to json file
        print("Add new conversation data into existing file")
        add_data_to_user_profile(profile_file, existing_conversation_data, hist_data, recorded_time, summary, pbi_data)
        
    else:
        # get the conversation summary
        summary = gen_text_inference(sum_model, sum_token, hist_data_text)
        # create new user profile
        print("Create new user profile")
        create_user_profile(user_data_path, new_profile, hist_data, recorded_time, summary, pbi_text)
        

if __name__ == "__main__":
    main()