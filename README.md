# CSYE7380-Soros

Local Installation
------------------

### Prerequisites

To run `app.py`, you'll need:
* [Python](https://www.python.org/downloads/)
    
### Installation
1. Install the required packages

```
pip install -r requirements.txt
```

2. Set up API keys (if you'd like to use OpenAI Chatbot):
   - Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
   - Add your OpenAI API key

3. Setup models and tokenizers
   - Download from https://drive.google.com/drive/folders/1WbeLh4gMiVXr48wn-Xn3mhmH0WY2gxV1
   - Place both model.h5 and tokenizer.subwords in `chatbot/models`

4. Run `app.py` on your local machine

```
streamlit run app.py
```
