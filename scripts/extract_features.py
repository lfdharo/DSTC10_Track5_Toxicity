"""
PURPORSE: This script takes a csv file with the following format:
                  id, toxic, answer
          And then extracts additional features such as avg. lenght, humour, sarcasm, NLI, Named-Entities substitutions, and emotions
AUTHOR:   Luis F. D'Haro
LICENSE:  GPLV3
"""


import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("file2process", type=str, help="The absolute path to the csv file to process. Must contain id, Toxic_turn, NonToxic_turnb")
parser.add_argument("-features", "--features", type=str, default="all", help="The features to extract, default:all, options: ratios, ner, humour, sarcasm, nli, emotions, perspective")
args = parser.parse_args()


# Check the required models exists
emotion_model = './models/best-emotional-model.pt'
humour_model = './models/colbert-trained/'

if os.path.exists(emotion_model) is False:
    print("The pre-trained model for emotion detection could not be found at {}. Exiting.".format(emotion_model))
    exit(0)
elif os.path.exists(humour_model) is False:
    print("The pre-trained model for humour detection could not be found at {}. Exiting.".format(humour_model))
    exit(0)
elif os.path.exists(args.file2process) is False:
    print("File {} could not be found. Exiting.".format(args.file2process))
    exit(0)
elif os.path.getsize(args.file2process) == 0:
    print("File {} is empty. Exiting.".format(args.file2process))
    exit(0)  



with open(args.file2process, 'r') as f_in:
    df_out = pd.read_csv(f_in)
    # Check the dataframe contains the columns we need
    if 'id' not in df_out.columns or 'toxic' not in df_out.columns or 'answer' not in df_out.columns: 
        print("The provided file {} does not contain the required columns: UID, toxic, answer. Exiting.".format(args.file2process))
        exit(0)

# Apply an additional filter over the generated dataset to avoid the answer to contain toxicity
# 320 words
swear_words = ["anal", "arse", "arsehole", "ass", "assbackwards", "asshole", "ass-hole", "assholes", "asskisser", "ass-lick", "asswipes", "bad-ass", "ballbreakers", 
            "barelylegal", "bastard", "basterds", "beater", "ben-wa", "bestiality", "bitch", "bitches", "bloody", "blow job", "blowjob", "blowjobs", "blowme", "bobo", 
            "bollock", "bollocks", "bondage", "boners", "boob", "boobie", "booby", "bozo", "bozos", "breast", "breasts", "brownnose", "bugga", "bugger", "bulldykes", "bullshit", 
            "bullshit", "bung", "butt", "buttwipe", "callgirl", "cameltoe", "candyass", "chickenshit", "chink", "chinks", "chinky", "choad", "clusterfuck", "cock", "cocksucker", 
            "cocksuckers", "coke", "cooch", "coochie", "coon", "coonass", "crap", "crappy", "craps", "cretin", "crikey", "cripple", "cum", "cums", "cunt", "dago", "dammit", 
            "damn", "damned", "damnit", "dick", "dickhead", "dickwad", "dildo", "dirty", "dogshit", "dominatrix", "donkeyshow", "doo-doo", "dope", "dork", "douche", "douchebag", 
            "dumbass", "dumb-ass", "dumber", "dumbfuck", "dumbshit", "dump", "dyke", "eatme", "fag", "faggot", "faggots", "fags", "fart", "farted", "farter", "farting", "farts", 
            "fatass", "felcher", "freak", "freaking", "fuck", "fucked", "fuckedup", "fucken", "fucker", "fuckers", "fuckface", "fuckhead", "fucking", "fuckoff", "fuckpigs", 
            "fucks", "fuckup", "fuckwit", "gangbang", "gang-bang", "gaybar", "gethorny", "getshorny", "gettinghorny", "godamn", "goddamn", "goddamni", "goddamnit", "gook", 
            "gooks", "gotohell", "halfassed", "hardass", "hardon", "hard-on", "hell", "hoe", "honky", "hooker", "horniness", "horny", "horseshit", "humper", "hussy", "hyman", 
            "idiot", "injuns", "jackme", "jackoff", "jailbait", "jerk", "jewboy", "jiggy", "jism", "jiz", "junglebunny", "kike", "kikes", "kunt", "lesbo", "lesbos", "limey", 
            "lovejuice", "merde", "merkin", "mindfuck", "mongoloid", "monkey", "moron", "morons", "motherfucker", "motherfuckers", "muff", "muthafucka", "mydick", "nad", 
            "nakedgirls", "nasty", "nekkid", "nig", "nigga", "niggas", "nigger", "niggers", "niggers", "nob", "nobs", "nookie", "nymphos", "one-eyedsnake", "orgies", "orgy", 
            "paki", "pakis", "paps", "pecker", "peckerwood", "pee", "peed", "peeing", "pee-pee", "pees", "penis", "perverts", "phonesex", "pig", "pillock", "piss", "pissed", 
            "pissedoff", "pissed-off", "pisser", "pissing", "polack", "polacks", "poo", "poon", "poontang", "poon-tang", "poos", "pooter", "porn", "prat", "pussies", "pussy", 
            "pussywhipped", "putz", "queer", "rape", "raped", "rat", "schlong", "schmuck", "schmucks", "scumbag", "scumbags", "semen", "shag", "shat", "shit", "shithead", 
            "shithole", "shithouse", "shitlist", "shits", "shitstains", "shitter", "shitting", "shitty", "shlong", "sicko", "skank", "skanky", "slag", "sleazebag", "sleaze-bag", 
            "sleazebags", "sleazeballs", "sleeparound", "sleptaround", "slut", "sluttier", "slutty", "sonofabitch", "spankyou", "spic", "spick", "spicks", "spics", "spunk", 
            "spunked", "striptease", "stupid", "sucker", "tadger", "tata", "ta-ta", "ta-tas", "tightass", "tit", "tits", "tittie", "titty", "todger", "topless", "tosser", 
            "turd", "twat", "upchuck", "urine", "vagina", "vibrators", "wank", "wanker", "wanker", "wenching", "wetbacks", "whack", "whitey", "whore", "whorehouse", "whoremaster", 
            "whore-monger", "whores", "wino", "winos", "wogs", "wop", "wops", "wuss", "wuss", "wussies", "yid"]
threatening_words = ['kill', 'suicide']

print('Removing turns where the answer is not appropriated')
df_out = df_out[~df_out['answer'].str.contains('|'.join(swear_words), case=False)]
df_out = df_out[~df_out['answer'].str.contains('|'.join(threatening_words), case=False)] 

filename_out = args.file2process.replace('.csv', '_new.csv')
with open(filename_out, 'w') as f_out:
    # Writing the new csv file
    print('Saving the filtered dataset')
    df_out.to_csv(f_out, index=False)


# Extract simple information as lenght of the sentences, ratios between
if 'ratios' in args.features or args.features == "all":
    print('**** Calculating RATIOS')
    if 'length_toxic' not in df_out.columns or 'length_answer' not in df_out.columns or 'ratio_lengths' not in df_out.columns:
        print('Performing extraction of lenght info')
        length_toxic = list()
        length_answer = list()
        ratio_lengths = list()
        for i, r in tqdm(df_out.iterrows()):
            a = len(r['toxic'].split())
            b = len(r['answer'].split())
            length_toxic.append(a)
            length_answer.append(b)
            ratio_lengths.append(b/a)
        df_out['length_toxic'] = length_toxic
        df_out['length_answer'] = length_answer
        df_out['ratio_lengths'] = ratio_lengths
        with open(filename_out, 'w') as f_out:
            # Writing the new csv file
            print('Saving the ratios info')
            df_out.to_csv(f_out, index=False)


# Extract the Sentiment information
if 'ner' in args.features or args.features == "all":
    print('**** Calculating NER')
    if 'answer_ner' not in df_out.columns or 'toxic_ner' not in df_out.columns:
        import stanza
        stanza.download('en')

        import spacy
        nlp = spacy.load("en_core_web_trf")

        max_entities_per_phrase = 2 # Max number of entites per phrase allowed, if not sentence is removed
        nlp = stanza.Pipeline(lang='en', processors='tokenize, ner')            
        for t in ['toxic', 'answer']:
            post_processed_sent = list()
            for _, row in tqdm(df_out.iterrows()):
                new_sentence = ""
                doc = nlp(row[t])
                num_entities_phrase = 0
                for sent in doc.sentences:
                    num_entities_phrase += len(sent.ents)
                    end_char = 0
                    for token in sent.tokens:
                        # num_entities_phrase = num_entities_phrase + 1
                        if "PERSON" in token.ner:
                            new_sentence = new_sentence + ' [PERSON] '
                        elif "NORP" in token.ner:
                            new_sentence = new_sentence + ' [NORP] '                               
                        elif "FAC" in token.ner:
                            new_sentence = new_sentence + ' [FACILITY] '
                        elif "ORG" in token.ner:
                            new_sentence = new_sentence + ' [ORG] '                            
                        elif "GPE" in token.ner:
                            new_sentence = new_sentence + ' [GPE] ' 
                        elif "LOC" in token.ner:
                            new_sentence = new_sentence + ' [LOC] '
                        elif "PRODUCT" in token.ner:
                            new_sentence = new_sentence + ' [PRODUCT] '
                        elif "EVENT" in token.ner:
                            new_sentence = new_sentence + ' [EVENT] '
                        elif "WORK_OF_ART" in token.ner:
                            new_sentence = new_sentence + ' [WORK_OF_ART] '
                        elif "LAW" in token.ner:
                            new_sentence = new_sentence + ' [LAW] '
                        elif "LANGUAGE" in token.ner:
                            new_sentence = new_sentence + ' [LANGUAGE] '                                                                                                                                                                       
                        elif "DATE" in token.ner:
                            new_sentence = new_sentence + ' [DATE] '
                        elif "TIME" in token.ner:
                            new_sentence = new_sentence + ' [TIME] '
                        elif "PERCENT" in token.ner:
                            new_sentence = new_sentence + ' [PERCENT] '                            
                        elif "MONEY" in token.ner:
                            new_sentence = new_sentence + ' [MONEY] '
                        elif "QUANTITY" in token.ner:
                            new_sentence = new_sentence + ' [QUANTITY] '
                        elif "CARDINAL" in token.ner:
                            new_sentence = new_sentence + ' [CARDINAL] '
                        elif "ORDINAL" in token.ner:
                            new_sentence = new_sentence + ' [ORDINAL] '                                                                                                                                                                                                                                                                     
                        else:
                            val = int(token.misc.split('|')[0].split('=')[1])
                            if end_char == val: # We have a word that must be joined to a previous one
                                new_sentence = new_sentence + token.text
                            else: # introduce a space
                                new_sentence = new_sentence + ' ' + token.text  
                            end_char = int(token.misc.split('|')[1].split('=')[1])
                if num_entities_phrase < max_entities_per_phrase:
                    doc = nlp(new_sentence)
                    new_sentence = ' '.join([token.text for token in doc])
                    post_processed_sent.append(new_sentence)
                else:
                    post_processed_sent.append("N.A.")
            df_out['{}_ner'.format(t)] = post_processed_sent
            df_out = df_out[df_out['{}_ner'.format(t)] != "N.A."] # Remove those that were not useful
        
        with open(filename_out, 'w') as f_out:
            # Writing the new csv file
            print('Saving the NRE info')
            df_out.to_csv(f_out, index=False)        


# Extract the humour information
if 'humour' in args.features or args.features == "all":
    print('**** Calculating HUMOUR')
    import humour
    if 'humour_score' not in df_out.columns:
        inputs = humour.compute_input_arrays(df_out, 'answer')
        preds = humour.predict(inputs)
        df_out['answer_humour_score'] = preds

        inputs = humour.compute_input_arrays(df_out, 'toxic')
        preds = humour.predict(inputs)
        df_out['toxic_humour_score'] = preds

        with open(filename_out, 'w') as f_out:
            # Writing the new csv file
            print('Saving the humour info')
            df_out.to_csv(f_out, index=False)


if 'sarcasm'  in args.features or args.features == "all":
    print('**** Calculating SARCASM')
    if 'sarcasm' not in df_out.columns:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelWithLMHead
        tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-sarcasm-twitter")
        model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-sarcasm-twitter")


        def eval_conversation(text):
            input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')
            output = model.generate(input_ids=input_ids, max_length=3)
            dec = [tokenizer.decode(ids) for ids in output]
            label = dec[0]
            return label


        answer_sarcasm = list()
        toxic_sarcasm = list()
        for i, r in tqdm(df_out.iterrows()):
            label = eval_conversation(r['answer'])
            if label == 'derison':
                answer_sarcasm.append(1)
            else:
                answer_sarcasm.append(0)

            label = eval_conversation(r['toxic'])
            if label == 'derison':
                toxic_sarcasm.append(1)
            else:
                toxic_sarcasm.append(0)

        df_out['answer_sarcasm'] = answer_sarcasm
        df_out['toxic_sarcasm'] = toxic_sarcasm
        with open(filename_out, 'w') as f_out:
            # Writing the new csv file
            print('Saving the sarcasm info')
            df_out.to_csv(f_out, index=False)

# Extract NLI information
if 'nli' in args.features or args.features == "all":
    print('**** Calculating NLI')
    if 'contradiction' not in df_out.columns or 'neutral' not in df_out.columns or 'entailment' not in df_out.columns:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelWithLMHead

        # Perform NLI task
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli")
        contradiction = list()
        neutral = list()
        entailment = list()
        for i, r in tqdm(df_out.iterrows()):
            inputs = tokenizer('[CLS]' + r['toxic'] + '[SEP] ' + r['answer'] + '[SEP]', return_tensors="pt")
            outputs = model(**inputs)
            scores = outputs.logits.softmax(dim=-1).detach().numpy()[0]
            contradiction.append(scores[0])
            neutral.append(scores[1])
            entailment.append(scores[2])

        df_out['contradiction'] = contradiction
        df_out['neutral'] = neutral
        df_out['entailment'] = entailment
        with open(filename_out, 'w') as f_out:
            # Writing the new csv file
            print('Saving the NLI info')
            df_out.to_csv(f_out, index=False)


# Extract the Sentiment information
if 'emotions' in args.features or args.features == "all":
    print('**** Calculating EMOTIONS')
    if 'toxic_emotion' not in df_out.columns or 'answer_emotion' not in df_out.columns or 'toxic_emotion_score' not in df_out.columns or 'answer_emotion_score' not in df_out.columns:
        from flair.models import TextClassifier
        from flair.data import Sentence

        # Works only with Transformers 3.5.0
        classifier = TextClassifier.load(emotion_model)

        def emotion_prediction(txt):
            # create example sentence
            sentence = Sentence(txt)
            # predict class and print
            classifier.predict(sentence)
            return sentence.labels[0]
        
        def handle_message(msg):
            """
            Emotions: anger, disgust, fear, happiness, neutral, sadness and surprise
            """
            return emotion_prediction(msg)

        toxic_emotion = list()
        toxic_emotion_score = list()
        answer_emotion = list()
        answer_emotion_score = list()
        for _, msg in tqdm(df_out.iterrows()):
            outputs_toxic = handle_message(str(msg['toxic']))
            toxic_emotion.append(outputs_toxic.value)
            toxic_emotion_score.append(outputs_toxic.score)
            outputs_answer = handle_message(str(msg['answer']))
            answer_emotion.append(outputs_answer.value)
            answer_emotion_score.append(outputs_answer.score)
        df_out['toxic_emotion'] = toxic_emotion
        df_out['toxic_emotion_score'] = toxic_emotion_score
        df_out['answer_emotion'] = answer_emotion
        df_out['answer_emotion_score'] = answer_emotion_score

        with open(filename_out, 'w') as f_out:
            # Writing the new csv file
            print('Saving the emotion info')
            df_out.to_csv(f_out, index=False)


# Extract the Sentiment information
if 'perspective' in args.features or args.features == "all":
    print("**** Calculating PERSPECTIVE")
    if 'toxic_perspective_score' not in df_out.columns or 'answer_perspective_score' not in df_out.columns:
        from googleapiclient import discovery
        import time

        print("CALCULATING TOXICITY WITH PERSPECTIVE")
        API_KEY = 'API_KEY_MUST_BE_PROVIDED'

        client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

        # Store the calculated scores
        toxic_perspective_score = list()
        answer_perspective_score = list()

        # If you are starting over and have already collected perspective scores then load them
        filename_perspective_scores = filename_out.replace('.csv', '.npz')
        if os.path.exists(filename_perspective_scores) is True and os.path.getsize(filename_perspective_scores) > 0:
            with open(filename_out.replace('.csv', '.npz'), 'rb') as f_out:
                print('Reading the perspective info from {}'.format(filename_perspective_scores))
                data = np.load(f_out, t=toxic_perspective_score, a=answer_perspective_score)
                toxic_perspective_score = data['t']
                answer_perspective_score = data['a']

        len_pre_calculated = len(toxic_perspective_score)
        for num, msg in tqdm(df_out.iterrows()):

            # Skip those that were already calculated
            if num < len_pre_calculated:
                continue
            try:
                analyze_request = {
                    'comment': {'text': "{}".format(msg["toxic"])},
                    'requestedAttributes': {'TOXICITY': {}}
                }

                response = client.comments().analyze(body=analyze_request).execute()
                v = response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
                toxic_perspective_score.append(v)
                time.sleep(1)  # Perspective requieres the API process one request per second
            except:
                toxic_perspective_score.append('NaN')
            try:
                analyze_request = {
                    'comment': {'text': "{}".format(msg["answer"])},
                    'requestedAttributes': {'TOXICITY': {}}
                }
                response = client.comments().analyze(body=analyze_request).execute()
                v = response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
                answer_perspective_score.append(v)
                time.sleep(1)  # Perspective requieres the API process one request per second
            except:
                answer_perspective_score.append('NaN')

            if (num % 60) == 0:
                with open(filename_out.replace('.csv', '.npz'), 'wb') as f_out:
                    # Writing the new csv file
                    print('Saving the perspective info for {}'.format(str(num)))
                    np.savez(f_out, t=toxic_perspective_score, a=answer_perspective_score)

        df_out['toxic_perspective_score'] = toxic_perspective_score
        df_out['answer_perspective_score'] = answer_perspective_score
        with open(filename_out, 'w') as f_out:
            # Writing the new csv file
            print('Saving the emotion info')
            df_out.to_csv(f_out, index=False)
                    
