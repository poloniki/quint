# Quint

<p align="center">
  <img src="frontend/logo.png" alt="Logo">
</p>

<p align="center">
  <a href="#">
    <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  </a>
</p>

"Quint" is designed to enhance the podcast experience. It simplifies the process for users, making it easier for them to understand and navigate podcasts by providing concise summaries, highlights, and transcripts.

## 🚀 Main Functionality

Below is a list of the core API endpoints offered by Quint:

### 🎥 YouTube Video Transcription

Simply provide a YouTube video ID. Quint will fetch the video, extract its audio content, and return a transcription of the audio.

`GET /youtube_transcript?video_id=YOUR_YOUTUBE_VIDEO_ID`

### 🎙️ Transcription from Audio File

Upload an audio file and instantly receive its transcription in text format.

`POST /file_transcript`

### 📜 Text Chunking

Submit a lengthy text and get it divided into semantically meaningful chunks or paragraphs.

`POST /chunk`

### 🌟 Highlight the Best Sentences

Submit a text and let Quint analyze it. The endpoint returns the index of the most descriptive sentence based on the embeddings.

`POST /best_sentence`

### 📖 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

## 🛜 How to deploy this API on cloud

Importand note: I highly reccomend using the JAX solution, as it is much faster than OpenAI proposed way. Please refer to this git [Whisper Jax](https://github.com/sanchit-gandhi/whisper-jax) for more details. I will attach one of the table from this git repo:

**Table 1:** Average inference time in seconds for audio files of increasing length. GPU device is a single A100 40GB GPU.
TPU device is a single TPU v4-8.

<div align="center">

|           | OpenAI  | Transformers | Whisper JAX | Whisper JAX |
| --------- | ------- | ------------ | ----------- | ----------- |
|           |         |              |             |             |
| Framework | PyTorch | PyTorch      | JAX         | JAX         |
| Backend   | GPU     | GPU          | GPU         | TPU         |
|           |         |              |             |             |
| 1 min     | 13.8    | 4.54         | 1.72        | 0.45        |
| 10 min    | 108.3   | 20.2         | 9.38        | 2.01        |
| 1 hour    | 1001.0  | 126.1        | 75.3        | 13.8        |
|           |         |              |             |             |

</div>

### Choosing a cloud provider

Of course you are free to choose any cloud provider as Google/Azure/AWS. However, one of the issues for me is that thoose providers do not have ADA6000 GPU's which are cheaper then A100 and likes and provide the same [compute capability](https://developer.nvidia.com/cuda-gpus). For that reason I propose - [Datacrunch](https://datacrunch.io/). This is not an advertisement, I'm in no way affiliated to them, but I found their service as best for cost/price for transcribtion.

1. We need to create an account via Sign-Up
2. Add card and top-up minumum ammount (20USD currently)
3. In the account settings Generate credentials and fill in .env.sample with Id and Secret
4. Rename .env.sample to .env
5. Run direnv reload in terminal

### Connect to DataCrunch client

```python
import os
from datacrunch import DataCrunchClient

# Get client secret from environment variable

CLIENT_SECRET = os.environ['DATACRUNCH_CLIENT_SECRET']
CLIENT_ID = os.environ['DATACRUNCH_CLIENT_ID']

# Create datcrunch client

datacrunch = DataCrunchClient(CLIENT_ID, CLIENT_SECRET)
```

### Create an SSH key and save it in ~/.ssh folder

```python
import os
import subprocess
import shutil
email = os.environ['EMAIL']
key_filename = email.replace("@", "_").replace(".", "_")
def generate_ssh_key(email: str) -> str:
    try:
        # Create a filename based on the email


        # Define the command and its arguments as a list
        cmd = [
            "ssh-keygen",
            "-t", "ed25519",
            "-C", email,
            "-f", key_filename,  # Using the modified filename
            "-N", ""  # Empty passphrase
        ]

        # Run the command
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Send an empty passphrase (newline) twice: once for the passphrase itself and once to confirm it
        process.communicate(input=b'\n\n')
        if process.returncode != 0:
            raise Exception("Error generating the key.")

        # Read the public key
        with open(f"{key_filename}.pub", "r") as f:
            pubkey = f.read()

        # Move generated keys to the ~/.ssh folder
        shutil.move(key_filename, os.path.expanduser(f"~/.ssh/{key_filename}"))
        shutil.move(f"{key_filename}.pub", os.path.expanduser(f"~/.ssh/{key_filename}.pub"))

        print(f"SSH key for {email} generated successfully!")
        return pubkey

    except Exception as e:
        print(f"Failed to generate SSH key for {email}. Reason: {e}")
        return None

# Get all SSH keys
ssh_keys = datacrunch.ssh_keys.get()
ssh_keys = list(map(lambda key: key.id, ssh_keys))
if len(ssh_keys) == 0:
    public_key = generate_ssh_key(email)
    datacrunch.ssh_keys.create(email,public_key)
    ssh_keys = datacrunch.ssh_keys.get()
    ssh_keys = list(map(lambda key: key.id, ssh_keys))

```

### Select GPU (by default A6000, but you can change to any available on website)

```python
#We select accessible GPU's that work really fast
instance_type=''
if datacrunch.instances.is_available(instance_type='1A6000.10V'):
    instance_type='1A6000.10V'
elif datacrunch.instances.is_available(instance_type='1A6000ADA.10V'):
    instance_type='1A6000ADA.10V'
```

### Create script for installing neccesary NVIDIA drivers

Edit path to script according to enviroment you are in. Otherwise you can just run the notebook.

```python
#Read the Bash script's contents
with open('../scripts/clone_github_code.sh', 'r') as file:
    script_contents = file.read()

#Use the script_contents in the datacrunch function
datacrunch.startup_scripts.create('cuda_drivers', script_contents)
script_id = datacrunch.startup_scripts.get()[0].id
```

### Create the new Instance

```python
#Create a new instance
instance = datacrunch.instances.create(instance_type=instance_type,
                                      image='ubuntu-22.04-cuda-12.0-docker',
                                      ssh_key_ids=ssh_keys,
                                      hostname='quint',
                                      description='Transcription and summarization API',
                                      startup_script_id=script_id,
                                      os_volume={
                                        "name": "OS volume",
                                        "size": 100
                                        }
                                      )

import time
while datacrunch.instances.get()[0].status=='provisioning':
    time.sleep(5)
print('Instance is running, you can now connect to it.')

```

### Connect to instance via terminal

```python
host = datacrunch.instances.get()[0].ip
print(f'ssh root@{host} -i ~/.ssh/{key_filename}')
Copy this command and run in terminal to connect to the instance.
```

#Connect to server via ssh and perform command "sudo reboot" to restart server for enabling Nvidia Drivers

```shell
sudo reboot
```

Inside of instance terminal run this commands to copy latest version

```shell
apt install gh
git clone https://github.com/poloniki/quint.git
```

Build and run docker

```shell
docker build -t quint --file Dockerfile.jax .
docker run --gpus all -p 80:80 quint
```

Your API endpoint would be avialable on the instance Ip-adress.
