from os import access
import requests
from utils import DEFAULT_PUSHBULLET_ACCESS_TOKEN, SESSION_ID


def send_pushbullet_message(message, title=f'CIL 2022 Project (#{SESSION_ID})',
                            access_token=DEFAULT_PUSHBULLET_ACCESS_TOKEN):
    if None in [message, access_token]:
        return
    try:
        requests.post('https://api.pushbullet.com/v2/pushes',
                      json={'type': 'note', 'title': title, 'body': message + '\n'},
                      headers={'Content-Type': 'application/json', 'Access-Token': access_token})
    except:
        pass
