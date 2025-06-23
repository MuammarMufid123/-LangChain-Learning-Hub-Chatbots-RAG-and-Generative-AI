import validators
import streamlit as st
import traceback

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# ─── Streamlit page configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website",
    page_icon="🦜",
    layout="wide",
)
st.title("🦜 LangChain: Summarize Text From YouTube or Website")
st.subheader("Enter a URL to generate a 300-word summary")

# ─── Sidebar: Groq API key ───────────────────────────────────────────────────────
groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    value="",
    type="password",
    help="Get your API key from https://console.groq.ai",
)

# ─── Main input: URL ────────────────────────────────────────────────────────────
generic_url = st.text_input(
    "URL (YouTube or Website)",
    placeholder="https://www.youtube.com/watch?v=...",
    label_visibility="visible",
)

# ─── Prepare the prompt template ────────────────────────────────────────────────
prompt = PromptTemplate(
    template=(
        "Provide a summary of the following content in 300 words:\n\n"
        "Content:\n"
        "{text}"
    ),
    input_variables=["text"],
)

# ─── Button trigger ─────────────────────────────────────────────────────────────
if st.button("Summarize"):
    # 1) Basic validation
    if not groq_api_key.strip():
        st.error("❌ Please enter your Groq API key in the sidebar.")
    elif not generic_url.strip():
        st.error("❌ Please enter the URL you want to summarize.")
    elif not validators.url(generic_url):
        st.error("❌ That doesn’t look like a valid URL. Please check and try again.")
    else:
        try:
            with st.spinner("🔍 Loading content and generating summary..."):
                # 2) Load documents based on URL type
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url, add_video_info=True
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        # Turn SSL verify on unless you have specific cert issues:
                        ssl_verify=True,
                        headers={
                            "User-Agent": (
                                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/116.0.0.0 Safari/537.36"
                            )
                        },
                    )
                docs = loader.load()

                # 3) Instantiate the LLM and summarization chain
                llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)
                chain = load_summarize_chain(
                    llm, chain_type="stuff", prompt=prompt
                )

                # 4) Run the summarization
                summary = chain.run(docs)
                st.success("✅ Summary generated successfully!")
                st.write(summary)

        except Exception as e:
            # Display the full traceback for debugging
            st.error("⚠️ An error occurred during summarization:")
            st.exception(e)
            st.write(
                "Please check:\n"
                "• That your API key is correct and has sufficient quota\n"
                "• The URL is reachable and valid\n"
                "• Your network connection"
            )
