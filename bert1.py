import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import base64

# Function to set dynamic styles based on user input
def set_dynamic_styles(text_color, font_size):
    css = f"""
    <style>
    .stApp .block-container {{
        margin-left: 0 !important;  /* Align content to the left */
        padding-left: 20px;  /* Add padding from the left */
        text-align: left;  /* Ensure text is left-aligned */
        color: {text_color};  /* Set text color */
        font-size: {font_size}px;  /* Set font size */
    }}
    h1, h2, h3, h4, h5, h6 {{
        text-align: left;  /* Align titles to the left */
        color: {text_color};  /* Set title color */
    }}
    .stTextInput > div {{
        margin-left: 0 !important;  /* Align the input box */
    }}
    .stButton > button {{
        margin-left: 0 !important;  /* Align the button */
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Function to set background image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}"); 
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Call the function to set the background (you need to provide the correct image path)
set_background("background.avif")

# Sidebar for style customization
st.sidebar.title("Customize Chatbot Style")

# Sidebar widgets to customize styles
text_color = st.sidebar.color_picker("Choose Text Color", "#000000")
font_size = st.sidebar.slider("Choose Font Size", 10, 30, 14)

# Apply the dynamic styles to the app
set_dynamic_styles(text_color, font_size)

# Sidebar for chatbot settings
st.sidebar.title("Chatbot Settings")

# Select question category from the sidebar
category = st.sidebar.selectbox("Select a Category", ["General", "Artificial intelligence", "Programming", "Cloud Computing"])

# Load BERT tokenizer and model
@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

tokenizer, model = load_bert_model()

# Predefined questions and responses for different categories
qa_pairs = {
    "General":
     {
        "What is your name?": "I am a chatbot powered by BERT!",
        "How are you?": "I'm just a bunch of code, but I'm doing great!",
    },
    "Artificial intelligence":
     {
        "What is AI?": "Artificial Intelligence (AI) is the ability of machines to simulate human intelligence. It enables systems to learn from data, make decisions, and perform tasks like understanding language, recognizing images, or solving problems.",
        "What is BERT?": "BERT stands for Bidirectional Encoder Representations from Transformers. It’s a powerful NLP model.",
        "What are Neural Networks?":"Neural Networks are algorithms modeled after the human brain's structure. They consist of layers of nodes (neurons) connected by weighted edges, where each node processes input data and passes the result to the next node. Neural networks are used in deep learning to solve complex tasks.",
    },
    "Programming": 
    {
        "What is Git?": "Git is a version control system that helps developers track and manage changes to code. It's widely used for collaboration in software development.",
        "What is Python?": "Python is a high-level programming language known for its simplicity and versatility. It's widely used in data science, web development, and automation.",
        "what is java programming":"Java is a high-level, class-based, object-oriented programming language that is designed to have as few implementation dependencies as possible.",
    },
    "Cloud Computing": 
    {
        "What is Cloud Computing?": "Cloud computing refers to delivering computing services—like storage, processing, and software—over the internet. Examples include AWS, Azure, and Google Cloud.",
        "What is Microsoft Azure?": "Microsoft Azure is a cloud computing platform and service provided by Microsoft. It offers tools and resources for building, deploying, and managing applications and services through Microsoft's global data centers.",
    }
}

# Function to get BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=130)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Precompute embeddings for predefined questions
@st.cache_resource
def compute_predefined_embeddings():
    return {question: get_bert_embedding(question) for category in qa_pairs for question in qa_pairs[category]}

predefined_embeddings = compute_predefined_embeddings()

# Function to get the chatbot's response based on category and user input
def chatbot_response(user_input, category):
    user_embedding = get_bert_embedding(user_input)
    similarities = {
        question: cosine_similarity(user_embedding, predefined_embeddings[question])[0][0]
        for question in qa_pairs[category]
    }
    best_match = max(similarities, key=similarities.get)
    
    if similarities[best_match] > 0.5:  # Threshold can be adjusted
        return qa_pairs[category][best_match]
    else:
        return "I'm not sure how to respond to that."

# Streamlit Main Content
st.title("BERT Chatbot")
st.write("Welcome to the BERT-powered chatbot! Select your style and category from the sidebar, and ask me anything!")
st.subheader(f"Category: {category}")
st.write("Ask me a question related to your selected category.")

# User input field
user_input = st.text_input("You:", placeholder="Type your message here...")

# Display the response
if user_input:
    response = chatbot_response(user_input, category)
    st.write(f"**Chatbot:** {response}")

# Footer
st.markdown("---")
