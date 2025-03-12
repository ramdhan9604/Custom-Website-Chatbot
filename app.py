import streamlit as st
from helper import load_documents, split_text, create_vector_store, query_llm

def main():
    st.title("Custom Website Chatbot")
    
    # User input for URLs
    urls = st.text_area("Enter URLs (comma-separated):").split(',')
    
    if st.button("Process URLs"):
        with st.spinner("Loading documents..."):
            documents = load_documents(urls)
        
        with st.spinner("Splitting text..."):
            text_chunks = split_text(documents)
        
        with st.spinner("Creating vector store..."):
            vector_store = create_vector_store(text_chunks)
        
        st.session_state["vector_store"] = vector_store
        st.success("Processing complete! You can now ask questions.")
    
    # Query input
    query = st.text_input("Ask a question:")
    if query and "vector_store" in st.session_state:
        with st.spinner("Fetching answer..."):
            response = query_llm(query, st.session_state["vector_store"])
        
        st.write("### Answer:")
        st.write(response.get("answer", "No answer found."))
        
        # if "sources" in response:
        #     st.write("### Sources:")
        #     for source in response["sources"]:
        #         st.write(f"- {source}")

if __name__ == "__main__":
    main()
