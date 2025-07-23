# HDocReader_app

This is Flask-based web-app for transcribing handwritten text entered by a user or from an uploaded document at the line or paragraph level. 

The app has two components: 
1) **TrOCR** : This pre-trained encoder-decoder model performs a first draft of the transcription from the uploaded document. 
2) **Gemini-1.5-flash** : This LLM corrects the first draft using the image of the document as context.

   

<img width="800" height="600" alt="svg_code" src="https://github.com/user-attachments/assets/8643a6e8-a298-49bb-85b7-d8f58eb4cc9d" />




The app can be run locally using Docker (see the Dockerfile for details). You will need to have a google API key for calling the Gemini model. 

On Colab, the app can be run in a public url using **pyngrok**. After forking and importing the repo, first install the required packages:

```
!pip install -q -r requirements.txt
```

After signing up on the Ngrok website and generating your authorization token: 

```
from pyngrok import ngrok
NGROK_AUTH_TOKEN = "your_token"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
```
The public url for the app can then be generated using the command:
```
%run ./HDocReader_app/main.py
```
