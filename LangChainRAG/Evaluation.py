import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from sklearn.metrics import precision_score, recall_score, f1_score

CHROMA_PATH = "chroma"
OPENAI_API_KEY = ""

predefined_queries = {
    "Jaké jsou základní způsoby výuky na Ostravské univerzitě?": [
        "Základní způsoby výuky na OU jsou přednášky, cvičení, semináře, exkurze, praxe, speciální kurzy, konzultace, tutoriály, off-line a on-line výuka prostřednictvím Learning Management System."
    ],
    "Jaký je účel přednášek na Ostravské univerzitě?": [
        "V přednáškách se studující seznamují s poznatky a metodami příslušného vědního nebo uměleckého oboru, uvádějí se do samostatného studia a připravují se k aplikacím přednášené látky."
    ],
    "Jaký je účel cvičení na Ostravské univerzitě?": [
        "Ve cvičeních si studující upevňují a prohlubují vědomosti získané z přednášek a samostatného studia, aplikují teoretické poznatky a rozvíjejí dovednosti nebo realizují stanovený rozsah praktických úkolů."
    ],
    "Jaký je účel seminářů na Ostravské univerzitě?": [
        "V seminářích se za aktivní účasti studujících teoreticky a metodicky rozvíjejí a prohlubují poznatky předmětů."
    ],
    "Jaký je účel praxe na Ostravské univerzitě?": [
        "Praxe slouží k upevňování a rozšiřování vědomostí a dovedností získaných studiem na základě práce studujících na odpovídajícím pracovišti. Praxe je součástí studia a způsob její organizace, zajištění a provádění je dán opatřením rektora nebo děkana nebo pokyny příslušného zaměstnance katedry nebo fakulty."
    ],
    "Jaký je účel exkurzí na Ostravské univerzitě?": [
        "Exkurze slouží zejména k tomu, aby si studující v reálných podmínkách ověřovali teoretické vědomosti získané výukou některých předmětů a seznamovali se s metodami práce v praxi."
    ],
    "Jaký je účel speciálních kurzů na Ostravské univerzitě?": [
        "Speciální kurzy slouží zejména k získání vědomostí a praktických dovedností ve speciálně vybrané oblasti studia."
    ],
    "Jaký je účel konzultací na Ostravské univerzitě?": [
        "Konzultace pomáhají studujícímu při samostatném studiu. Konzultace může být individuální nebo skupinová. Její poskytování je vždy podmíněno aktivní účastí studujícího v ostatních způsobech výuky předmětu, pokud jsou stanoveny. Konzultací mohou být výjimečně nahrazeny i některé způsoby a části výuky."
    ],
    "Jaký je účel tutoriálů, off-line a on-line výuky prostřednictvím LMS na Ostravské univerzitě?": [
        "Tutoriály, off-line a on-line výuka prostřednictvím Learning Management System jsou primárními způsoby výuky u distanční, nebo kombinované formy studia."
    ],
    "V jakém jazyce probíhá výuka a kontroly studia na Ostravské univerzitě?": [
        "Výuka a kontroly studia probíhají v českém jazyce, to neplatí u studia z oblasti cizích jazyků."
    ],
    "V jakém jazyce mohou být vybrané předměty nebo jejich části vyučovány na Ostravské univerzitě?": [
        "Výuka a kontroly studia podle odstavce 10 u vybraných jednotlivých předmětů nebo jejich částí může probíhat též v cizím jazyce. Tato informace musí být zveřejněna v popisu předmětu ve veřejné části internetových stránek nejpozději poslední den lhůty stanovené pro podávání přihlášek. Postup dle čl. 5 odst. 5 se nepoužije."
    ],
    "Jaký je jazyk výuky a kontroly studia u programů akreditovaných v cizím jazyce na Ostravské univerzitě?": [
        "Jazyk výuky a kontrola studia u studijního programu akreditovaného v cizím jazyce je ten cizí jazyk, v kterém je studijní program akreditován."
    ]
}


def evaluate_retrieval(true_relevant_docs, retrieved_docs):
    true_binary = [1 if any(true_doc in doc for true_doc in true_relevant_docs) else 0 for doc in retrieved_docs]
    retrieved_binary = [1] * len(retrieved_docs)

    precision = precision_score(true_binary, retrieved_binary, zero_division=0)
    recall = recall_score(true_binary, retrieved_binary, zero_division=0)
    f1 = f1_score(true_binary, retrieved_binary, zero_division=0)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

def main():
    # Load the Chroma vector store
    embedding_function = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Evaluate each predefined query
    for query, true_relevant_docs in predefined_queries.items():
        print(f"\nQuery: {query}")

        # Search the DB
        results = db.similarity_search_with_relevance_scores(query, k=3)
        retrieved_docs = [doc.page_content for doc, _score in results]

        if not results:
            print("No documents retrieved for the query.")
            continue

        # Print detailed debug information
        for idx, (doc, score) in enumerate(results):
            print(f"Retrieved Doc {idx+1}: {doc.page_content[:200]}... with score {score}")

        # Evaluate retrieval
        print(f"Retrieved Docs: {retrieved_docs}")
        evaluate_retrieval(true_relevant_docs, retrieved_docs)

if __name__ == "__main__":
    main()
