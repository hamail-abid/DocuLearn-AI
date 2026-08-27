import streamlit as st
import requests

st.set_page_config(page_title="DocuLearn AI", page_icon="📚", layout="centered")

st.title("📚 DocuLearn AI ")
st.caption("A Smart RAG-based FYP Assistant for Interactive Document Learning")

BACKEND_URL = "http://127.0.0.1:8000"

# Sidebar: File Upload
st.sidebar.header("Document Upload Center")
uploaded_file = st.sidebar.file_uploader(
    "Upload your document here (PDF, DOCX, PPTX, TXT)", 
    type=["pdf", "docx", "pptx", "txt"]
)

if uploaded_file is not None:
    if st.sidebar.button("Upload & Process PDF"):
        with st.spinner(""Processing and indexing PDF..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                res = requests.post(f"{BACKEND_URL}/upload-and-index/", files=files)
                if res.status_code == 200:
                    st.sidebar.success("PDF Successfully Upload & Indexed!")
                    st.session_state.messages = []
                else:
                    st.sidebar.error(f"Error: {res.json().get('detail')}")
            except Exception as e:
                st.sidebar.error(f"Backend connection error: {e}")

# Continuous Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "pages" in msg and msg["pages"]:
            st.caption(f"📍 **Source Page Numbers:** {', '.join(map(str, msg['pages']))}")

# User Question Input
if user_query := st.chat_input("Ask a question based on the PDF:"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing document..."):
            try:
                res = requests.post(
                    f"{BACKEND_URL}/ask/",
                    json={"question": user_query}
                )
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer")
                    pages = data.get("pages", [])

                    st.write(answer)
                    if pages:
                        st.caption(f"📍 **Source Page Numbers:** {', '.join(map(str, pages))}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "pages": pages
                    })
                else:
                    err = res.json().get("detail", "Error occurred")
                    st.error(err)
            except Exception as e:
                st.error(f"Backend error: {e}")
