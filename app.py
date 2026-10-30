import streamlit as st
from rag_pipeline import CollegeRAG

st.set_page_config(page_title="College Knowledge Assistant", page_icon="🎓", layout="wide")
st.title("College Knowledge Assistant")
st.caption("Ask questions about fictional college policies. Answers are grounded in retrieved documents and include source citations.")

@st.cache_resource
def load_rag():
    return CollegeRAG()

try:
    rag = load_rag()
except Exception as e:
    st.error(f"Could not initialize the RAG system: {e}")
    st.info("Check that Ollama is running and the configured embedding model is available.")
    st.stop()

with st.sidebar:
    st.header("Knowledge base")
    st.write(f"Indexed chunks: **{rag.chunk_count}**")
    st.write("Documents are loaded from the `documents/` folder.")
    if st.button("Rebuild knowledge base"):
        with st.spinner("Rebuilding index..."):
            try:
                rag.rebuild_index()
                st.cache_resource.clear()
                st.success("Index rebuilt. Refresh the page if needed.")
            except Exception as e:
                st.error(str(e))
    st.markdown("---")
    st.write("**Safety behavior**")
    st.write("- Answers must be supported by retrieved text.")
    st.write("- Low-relevance questions are declined.")
    st.write("- Contradictory evidence is shown as a conflict.")

question = st.text_input("Your question", placeholder="What is the minimum attendance requirement?")
if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Searching official sample documents and preparing an answer..."):
        result = rag.ask(question.strip())
    st.subheader("Answer")
    st.write(result["answer"])
    if result["status"] == "conflict":
        st.warning("Conflicting information was found. Confirm the current rule with the college office.")
    elif result["status"] == "insufficient":
        st.info("No sufficiently relevant evidence was found in the knowledge base.")
    if result["sources"]:
        st.subheader("Supporting sources")
        for i, source in enumerate(result["sources"], start=1):
            with st.expander(f'{i}. {source["source"]} — {source["page_label"]}'):
                st.write(source["text"])
                st.caption(f'Retrieval similarity: {source["similarity"]:.3f}')
