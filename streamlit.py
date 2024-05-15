import streamlit as st
from help_desk import HelpDesk
from streamlit_extras.app_logo import add_logo

# Add logo to the app
logo_path = "xpeng.svg"
st.image(logo_path, width=150)

st.title("Ask Xpeng Chatbot")

@st.cache_resource
def get_model():
    model = HelpDesk(new_db=True)
    return model

model = get_model()

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
if prompt := st.chat_input("How can I help you?"):
    # Add user's prompt to the conversation
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Get answer from the model
    print(f"Received user prompt: {prompt}")
    result, sources = model.retrieval_qa_inference(prompt)
    print(f"Result: {result}")
    print(f"Sources: {sources}")

    # Prepare the assistant's response
    if sources:
        response = f"{result}\n\n{sources}"
    else:
        response = "Sorry, I couldn't find any relevant information to answer your question."

    # Add the assistant's response to the conversation
    st.chat_message("assistant").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
