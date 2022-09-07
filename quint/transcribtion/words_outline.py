import spacy
#nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_lg")



def get_words(text):
    for doc in nlp.pipe([text]):
        mentioned = [ent.lemma_ for ent in doc.ents if ent.label_ in ['PERSON', 'PRODUCT', 'ORG', 'GPE', 'MONEY', 'CARDINAL']]
    return mentioned

def outline(df):
    df['names'] = df.sentence.apply(get_words)
    return df
