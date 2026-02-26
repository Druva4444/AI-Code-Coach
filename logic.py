from retriever import get_retriever
from llm import get_llm
from prompts import DEBUG_PROMPT, TRANSLATE_PROMPT, EXPLAIN_PROMPT
from langchain.chains import LLMChain

def process_query(choice, query):
    """
    Processes a code-related query using RAG.
    choice: "1" (Debug), "2" (Translate), "3" (Explain)
    query: The user's question or problem description.
    """
    retriever = get_retriever()
    llm = get_llm()

    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(query)
    
    # Format context with metadata (filename)
    context_parts = []
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown')
        context_parts.append(f"--- File: {source} ---\n{doc.page_content}")
    
    context = "\n\n".join(context_parts)

    # Select prompt based on choice
    if choice == "1":
        prompt = DEBUG_PROMPT
    elif choice == "2":
        prompt = TRANSLATE_PROMPT
    else:
        prompt = EXPLAIN_PROMPT

    # Run the chain
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(context=context, query=query)
    
    return result, docs
