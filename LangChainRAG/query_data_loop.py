from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly. 

Context information:
{context}

---

{history}
User: {question}
Assistant: 
"""

def main():
    # Prepare the DB.
    embedding_function = OpenAIEmbeddings(api_key="")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Initialize the chat model.
    model = ChatOpenAI(api_key="")

    # Start the chat loop.
    conversation_history = ""
    query_text = input("Enter your initial question: ")

    while True:
        # Search the DB.
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        if len(results) == 0 or results[0][1] < 0.7:
            print(f"Unable to find matching results.")
            return

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(history=conversation_history, context=context_text, question=query_text)
        print(f"Prompt:\n{prompt}")

        # Get the response from the model.
        response_text = model.invoke(prompt)
        conversation_history += f"User: {query_text}\nAssistant: {response_text}\n"

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        print(formatted_response)

        # Get the next query from the user.
        query_text = input("Enter your next question (or 'exit' to quit): ")
        if query_text.lower() == 'exit':
            break

if __name__ == "__main__":
    main()
