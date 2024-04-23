# Streamlit
# Use QARetrieval to find informations about the Octo Confluence
# Basic example with a improvementd:
# Add streaming
# Add Conversation history
# Optimize Splitter, Retriever,
# Try Open source models
import streamlit as st
from help_desk import HelpDesk
from streamlit_extras.app_logo import add_logo

add_logo = "xpeng.svg"
st.image(add_logo, width=150)

st.title("Ask Xpeng chatbot")

@st.cache_resource
def get_model():
    model = HelpDesk(new_db=True)
    return model


model = get_model()

# Streamlit
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input("How can I help you??"):
    # Add prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Get answer
    result, sources = model.retrieval_qa_inference(prompt)

    # Add answer and sources
    st.chat_message("assistant").write(result + "  \n  \n" + sources)
    st.session_state.messages.append(
        {"role": "assistant", "content": result + "  \n  \n" + sources}
    )
