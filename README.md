# HDocReader_app

This is Flask-based web-app for transcribing handwritten text entered by an user or from an uploaded document at the line or paragraph level. 
The app has two components: 
1) A TrOCR model : This model performs a first draft of the transcription from the uploaded document. 
2) A Gemini-1.5-flash model : This LLM corrects the first draft using the image of the document as context. 

The app can be run locally using the Dockerfile. 

On Colab, the app can be run using pyngrok. After forking and importing the repo, first install the required packages:
```!pip install -q -r requirements.txt'''
