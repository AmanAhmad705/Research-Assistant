from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

 
# Initialize hosted LLaMA 3 model using Groq
llm = ChatGroq(
    groq_api_key= "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",  # Replace with your real key
    model_name="llama3-70b-8192"       # or "llama3-8b-8192"
)

def generate_answer(question, context_chunks):
    # Combine context
    context = "\n\n".join(context_chunks)

    # Construct prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful academic assistant. Answer the user's question based only on the following context."),
        ("user", "Context:\n" + context + "\n\nQuestion:\n" + question)
    ])

    # Create a chain
    chain = prompt_template | llm

    # Run the chain
    response = chain.invoke({"input": question})
    
    
    return response.content
