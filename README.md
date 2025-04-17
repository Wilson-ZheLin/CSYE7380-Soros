# CSYE7380-Soros

**Try Our Live Demo Here:** [https://soros-csye7380.streamlit.app](https://soros-csye7380.streamlit.app)

Local Installation
------------------

### Prerequisites

To run both `app.py` and tensorflow model, you'll need:
* Python 3.9

### Installation

1.  Setup your acceleration environment Apple Silicon only:

```zsh
conda create -n chatbot python=3.9
conda activate chatbot
conda install -c apple tensorflow-deps -y
pip install tensorflow-macos==2.9.1
pip install tensorflow-metal==0.5.0
```

2. Install the required packages

```
pip install -r requirements.txt
```

3. Set up API keys (if you'd like to use OpenAI Chatbot):
   - Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
   - Add your OpenAI API key

4. Setup models and tokenizers
   - Download from https://drive.google.com/drive/folders/1WbeLh4gMiVXr48wn-Xn3mhmH0WY2gxV1
   - Place both model.h5 and tokenizer.subwords in `chatbot/models`


5. Run `app.py` on your local machine

```
streamlit run app.py
```
