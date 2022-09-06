import re
import string
import datetime

def get_timestamp(message ,text, length):
    # try:
    #     message_tr = ' '.join(message.strip().split(' ')[:length])
    #     pattern = re.compile(f"(\[\d+.\d+\]) [^[]*{message_tr}")
    #     last_index = re.search(pattern , text)
    #     print('it succeeds')
    #     return last_index.group(1)
    # except AttributeError:
    for i in range(length,0,-1):
        try:
            message_tr = ' '.join(message.strip().split(' ')[length-i:2*length-i])
            pattern = re.compile(f"(\[\d+.\d+\]) [^[]*{message_tr}")
            last_index = re.search(pattern , text)
            print('second attempt')
            return last_index.group(1)
        except AttributeError:
            continue

#Import that
def preprocessing(sentence):
    sentence = ''.join([each for each in sentence if each not in string.punctuation])
    return sentence

def timestamping(chunks, transcript):
    # Creating row text with printed in timestamps to loop through
    concatenated_text = ""
    lines = []
    starting = []
    for i in transcript:
        lines.append(preprocessing(i['text']))
        starting.append(i["start"])
        concatenated_text += ' ' + f'{[i["start"]]} ' + preprocessing(i['text'])



    # Get the timestamps
    final_dict = {}
    timestamps = []

    chunks = list(map(lambda chunk : chunk.replace('\"','')[:200] , chunks))
    for i, each in enumerate(chunks):


        timestamp = get_timestamp(preprocessing(each[0:200].lower()), concatenated_text.lower(), 3)


        timestamp = round(float(re.findall("\d+\.\d+", timestamp)[0]))
        timestamp = str(datetime.timedelta(seconds=timestamp))
        final_dict.update({i:{timestamp:each}})
        timestamps.append(timestamp)
    return timestamps
