import re
import string
import datetime

def get_timestamp(message ,text, length):
    try:
        message_tr = ' '.join(message.strip().split(' ')[:length])
        pattern = re.compile(f"(\[\d+.\d+\]) [^[]*{message_tr}")
        last_index = re.search(pattern , text)
        return last_index.group(1)
    except AttributeError:
        message_tr = ' '.join(message.strip().split(' ')[length:])
        pattern = re.compile(f"(\[\d+.\d+\]) [^[]*{message_tr}")
        last_index = re.search(pattern , text)
        return last_index.group(1)
