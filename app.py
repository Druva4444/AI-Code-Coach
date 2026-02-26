from retriever import get_retriever
from llm import get_llm
from prompts import DEBUG_PROMPT, TRANSLATE_PROMPT, EXPLAIN_PROMPT
from langchain.chains import LLMChain

def main():
    retriever = get_retriever()
    llm = get_llm()

    print("\nAI Code Coach")
    print("1) Debug Code")
    print("2) Translate Code")
    print("3) Explain Algorithm")

    choice = input("Choose (1/2/3): ").strip()
    query = input("Describe your problem: ")

    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs])

    if choice == "1":
        prompt = DEBUG_PROMPT
    elif choice == "2":
        prompt = TRANSLATE_PROMPT
    else:
        prompt = EXPLAIN_PROMPT

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(context=context, query=query)

    print("\n--- AI Response ---\n")
    print(result)

if __name__ == "__main__":
    main()