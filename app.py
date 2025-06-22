import streamlit as st
from ibm_watsonx_ai.foundation_models import Model

# --- IBM Credentials ---
api_key = st.secrets["ibm"]["api_key"]
project_id = st.secrets["ibm"]["project_id"]
base_url = st.secrets["ibm"]["base_url"]
model_id = "ibm/granite-13b-instruct-v2"

# --- Streamlit Page Config ---
st.set_page_config(page_title="Smart City Assistant", layout="centered", page_icon="🏠")

# --- Custom Styling ---
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #e0f7fa, #e1f5fe);
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .stChatMessage.user {
            background-color: #fff3e0;
            color: #3e2723;
            border-radius: 10px;
            padding: 8px;
            margin-bottom: 6px;
        }
        .stChatMessage.assistant {
            background-color: #e3f2fd;
            color: #0d47a1;
            border-radius: 10px;
            padding: 8px;
            margin-bottom: 6px;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
            border: 1px solid #90caf9;
        }
    </style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown("""
    <div class="main">
        <h1 style='text-align: center; color: #0d47a1;'>💬 Chat with Smart City Assistant</h1>
        <p style='text-align: center; color: #37474f;'>Ask anything about sustainability, smart policies, green infrastructure, or city planning!</p>
    </div>
""", unsafe_allow_html=True)

# --- Chat History ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- Render Chat ---
for i in range(0, len(st.session_state.chat_history), 2):
    with st.chat_message("user"):
        st.write(st.session_state.chat_history[i])
    if i + 1 < len(st.session_state.chat_history):
        with st.chat_message("assistant"):
            st.markdown(st.session_state.chat_history[i + 1])

# --- Input Box ---
user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.chat_history.append(user_input)
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                model = Model(
                    model_id=model_id,
                    credentials={"apikey": api_key, "url": base_url},
                    project_id=project_id,
                )

                prompt = f"""You are a helpful smart city assistant focused on sustainability and policy advice.
Provide responses as bullet points where helpful, using a friendly tone.
Input: {user_input}
Response:"""

                response = model.generate_text(
                    prompt=prompt,
                    params={
                        "max_new_tokens": 512,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "decoding_method": "sample",
                        "stop_sequences": ["<|endoftext|>", "User:"],
                    }
                )

                output = response['generated_text'] if isinstance(response, dict) and 'generated_text' in response else str(response)
                st.markdown(output)
                st.session_state.chat_history.append(output)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.chat_history.append("Sorry, I encountered an issue.")
