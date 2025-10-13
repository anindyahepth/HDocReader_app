# HDocReader_app

This is Flask-based web-app for transcribing handwritten text entered by a user or from an uploaded document at the line or paragraph level. 

The workflow has the following components: 
1) **Draft Transcription** :  Pre-trained **TrOCR** model performs a first draft of the transcription from the uploaded document. 
2) **LLM Editor** : **Gemini-2.5-flash** corrects the first draft using the image of the document as context, and produces the final transcript.
3) **Evaluation** : The final transcript is evaluated by a second LLM **(GPT-4o)** which gives the transcript an accuracy score based on the evaluated Character Accuracy Rate (CER).
4) **MLflow Tracking/Real-time Performance Dashboard**.

   

<img width="800" height="600" alt="svg_code" src="https://github.com/user-attachments/assets/8643a6e8-a298-49bb-85b7-d8f58eb4cc9d" />




The app can also be run on Colab in a public url using **pyngrok**. After forking and importing the repo, first install the required packages:

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
