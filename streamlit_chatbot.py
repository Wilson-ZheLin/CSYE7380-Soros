import streamlit as st
import pandas as pd
from difflib import get_close_matches
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 设置页面标题
st.set_page_config(page_title="Soros 投资问答 Chatbot")

# 加载模型和数据
@st.cache_resource
def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("./soros-model").to("cpu")
    tokenizer = AutoTokenizer.from_pretrained("./soros-model")
    return model, tokenizer

@st.cache_data
def load_qa_data():
    df = pd.read_csv("Soros-QNA-Pairs.csv")
    questions = df["Question"].dropna().tolist()
    answers = df["Answer"].dropna().tolist()
    return questions, answers

model, tokenizer = load_model()
questions, answers = load_qa_data()

# 回答逻辑
def get_finetuned_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def get_answer(user_input):
    match = get_close_matches(user_input, questions, n=1, cutoff=0.6)
    if match:
        return answers[questions.index(match[0])]
    else:
        return get_finetuned_response(user_input)

# Streamlit UI
st.title("💬 索罗斯投资理念问答机器人")

user_input = st.text_input("请输入你的问题（例如：索罗斯怎么看金融市场危机？）")

if user_input:
    with st.spinner("思考中..."):
        response = get_answer(user_input)
    st.markdown("**💡 Chatbot 回答：**")
    st.success(response)

st.markdown("---")
st.caption("本模型基于乔治·索罗斯的投资理念进行微调，仅供学习参考。")
