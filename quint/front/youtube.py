import requests
def video_name(id):
    url = "https://youtube138.p.rapidapi.com/video/details/"

    querystring = {"id":id,"hl":"en","gl":"US"}

    headers = {
        "X-RapidAPI-Key": "d59a70c161msh031072ca8ec19f2p119afejsnf4860e522f15",
        "X-RapidAPI-Host": "youtube138.p.rapidapi.com"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    return response.json()['title']
