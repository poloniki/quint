

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
        sentences = re.findall('.*?[.!\?]\s', doc)
        print('Loaded Text String')
        return sentences
    elif type(doc) is list:
        sentences = re.findall('.*?[.!\?]\s', doc)

        print('Data is loaded as a list')
        return sentences
