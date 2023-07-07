from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

# load in all text file
welcome = TextLoader("./introDocs/welcome.txt")
install = TextLoader("./introDocs/install.txt")
quickstart = TextLoader("./introDocs/quickstart.txt")
help = TextLoader("./introDocs/help.txt")
running = TextLoader("./introDocs/running.txt")

values = TextLoader("./codeSamples/values.txt")
variables = TextLoader("./codeSamples/variables.txt")
tableindx = TextLoader("./codeSamples/tableindexing.txt")
statements = TextLoader("./codeSamples/statements.txt")
operators = TextLoader("./codeSamples/operators.txt")
identifiers = TextLoader("./codeSamples/identifiers.txt")
functions = TextLoader("./codeSamples/functions.txt")
blocks = TextLoader("./codeSamples/blocks.txt")



# create corpus of documents and split
documents = running.load()+welcome.load()+install.load()+quickstart.load()+help.load()
documents += values.load()+variables.load()+tableindx.load()+statements.load()+operators.load()+identifiers.load()+functions.load()+blocks.load() 
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# add openAI embedding for LLM capabilities
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

# customize the parameters
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 4}), return_source_documents=True)

# get user query and run to obtain results
while (True):
    query = input("Ask a question: ")
    print("\n")
    result = qa({"query": query})

    # print the answer and document citations
    print(result["result"])
    print("\n")

    for x in result["source_documents"]:
        print(x)
        print("\n")

