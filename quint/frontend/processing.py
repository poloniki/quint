
def concatenate_lines(transcript):
    text = ""
    for i in transcript:
        text += ' ' + i['text']
    return text
