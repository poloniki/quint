

def get_text_data(path_to_txt):
    """_summary_

    Args:
        path_to_txt (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(path_to_txt) as f:
        doc = f.readlines()
    if type(doc) is str:
        doc = doc.replace("?", ".")
        sentences = doc.split('.')
        print('Loaded Text String')
        return sentences
    elif type(doc) is list:
        doc = doc[0].replace("?", ".")
        sentences = doc.split('.')

        print('Data is loaded as a list')
        return sentences
