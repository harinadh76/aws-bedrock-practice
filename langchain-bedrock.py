"""
LESSON 5: Bedrock with LangChain
=================================
Use LangChain's higher-level abstractions with Bedrock
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                   BEDROCK + LANGCHAIN                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   WHY USE LANGCHAIN WITH BEDROCK?                                           │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                      │   │
│   │   • Simpler API - less boilerplate code                             │   │
│   │   • Easy model switching - change one line to switch models         │   │
│   │   • Built-in features - memory, chains, agents, RAG                 │   │
│   │   • Ecosystem - works with LangGraph, LangSmith, etc.              │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   BEFORE (Raw Boto3):              AFTER (LangChain):                       │
│   ┌─────────────────────┐          ┌─────────────────────┐                  │
│   │ 15+ lines of code   │    vs    │ 3 lines of code     │                  │
│   │ JSON formatting     │          │ Simple .invoke()    │                  │
│   │ Manual parsing      │          │ Clean responses     │                  │
│   └─────────────────────┘          └─────────────────────┘                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")


# ============================================
# Basic LangChain + Bedrock
# ============================================

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

def basic_langchain_example():
    """Simple example using LangChain with Bedrock"""
    
    print("\n📘 Basic LangChain Example")
    print("-" * 40)
    
    # Create the model - that's it! Much simpler than boto3
    llm = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        region_name="us-east-1",
        model_kwargs={
            "temperature": 0.7,
            "max_tokens": 500
        }
    )
    
    # Simple invocation
    response = llm.invoke("What is AWS Bedrock in one sentence?")
    print(f"Response: {response.content}")
    
    return llm


# ============================================
# Using Messages
# ============================================

def messages_example():
    """Using different message types"""
    
    print("\n📘 Messages Example")
    print("-" * 40)
    
    llm = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        region_name="us-east-1"
    )
    
    # You can use message objects for more control
    messages = [
        SystemMessage(content="You are a helpful coding assistant. Be concise."),
        HumanMessage(content="What's a Python list comprehension?"),
    ]
    
    response = llm.invoke(messages)
    print(f"Response: {response.content}")


# ============================================
# Streaming with LangChain
# ============================================

def streaming_example():
    """Stream responses with LangChain"""
    
    print("\n📘 Streaming Example")
    print("-" * 40)
    
    llm = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        region_name="us-east-1",
        streaming=True  # Enable streaming
    )
    
    print("Streaming response: ", end="")
    
    for chunk in llm.stream("Write a haiku about cloud computing."):
        print(chunk.content, end="", flush=True)
    
    print("\n")


# ============================================
# Using Prompt Templates
# ============================================

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def prompt_template_example():
    """Use prompt templates for reusable prompts"""
    
    print("\n📘 Prompt Template Example")
    print("-" * 40)
    
    llm = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        region_name="us-east-1"
    )
    
    # Create a reusable prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that explains {topic} concepts."),
        ("human", "Explain {concept} in simple terms.")
    ])
    
    # Create a chain
    chain = prompt | llm
    
    # Use the chain with different inputs
    response = chain.invoke({
        "topic": "AWS",
        "concept": "S3 buckets"
    })
    
    print(f"Response: {response.content}")


# ============================================
# Conversation with Memory
# ============================================

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def conversation_with_memory():
    """Create a chatbot with memory using LangChain"""
    
    print("\n📘 Conversation with Memory")
    print("-" * 40)
    
    llm = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        region_name="us-east-1"
    )
    
    # Create prompt with message history placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Remember what the user tells you."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    
    # Store for chat histories (by session ID)
    store = {}
    
    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    
    # Wrap with message history
    with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    # Test the conversation
    session_id = "user-123"
    
    # First message
    response1 = with_history.invoke(
        {"input": "My name is Alice and I love Python."},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"User: My name is Alice and I love Python.")
    print(f"Assistant: {response1.content}\n")
    
    # Second message - should remember
    response2 = with_history.invoke(
        {"input": "What's my name and what do I love?"},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"User: What's my name and what do I love?")
    print(f"Assistant: {response2.content}")


# ============================================
# Using Different Models Easily
# ============================================

def switch_models_example():
    """Easily switch between different Bedrock models"""
    
    print("\n📘 Switching Models Example")
    print("-" * 40)
    
    models = {
        "Claude Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        "Claude Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
        # "Llama 3": "meta.llama3-8b-instruct-v1:0",  # Uncomment if you have access
    }
    
    prompt = "What is 2+2? Answer with just the number."
    
    for name, model_id in models.items():
        try:
            llm = ChatBedrock(
                model_id=model_id,
                region_name="us-east-1"
            )
            response = llm.invoke(prompt)
            print(f"{name}: {response.content.strip()}")
        except Exception as e:
            print(f"{name}: Error - {e}")


# ============================================
# Embeddings with Bedrock
# ============================================

from langchain_aws import BedrockEmbeddings

def embeddings_example():
    """Use Bedrock for text embeddings (useful for RAG)"""
    
    print("\n📘 Embeddings Example")
    print("-" * 40)
    
    # Create embeddings model
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        region_name="us-east-1"
    )
    
    # Embed a single text
    text = "AWS Bedrock is a fully managed service for foundation models."
    vector = embeddings.embed_query(text)
    
    print(f"Text: {text}")
    print(f"Embedding dimensions: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
    
    # Embed multiple texts
    texts = [
        "Machine learning is fascinating.",
        "I love programming in Python.",
        "Cloud computing is the future."
    ]
    
    vectors = embeddings.embed_documents(texts)
    print(f"\nEmbedded {len(vectors)} documents")


# ============================================
# Simple RAG Example
# ============================================

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

def simple_rag_example():
    """Simple RAG (Retrieval Augmented Generation) example"""
    
    print("\n📘 Simple RAG Example")
    print("-" * 40)
    
    # Create embeddings
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        region_name="us-east-1"
    )
    
    # Create some documents
    documents = [
        Document(
            page_content="Amazon Bedrock is a fully managed service that offers foundation models.",
            metadata={"source": "aws-docs"}
        ),
        Document(
            page_content="Bedrock supports models from Anthropic, Meta, Amazon, and others.",
            metadata={"source": "aws-docs"}
        ),
        Document(
            page_content="You can use Bedrock for text generation, embeddings, and image generation.",
            metadata={"source": "aws-docs"}
        ),
        Document(
            page_content="Bedrock provides enterprise security with VPC support and encryption.",
            metadata={"source": "aws-docs"}
        ),
    ]
    
    # Create vector store
    vectorstore = InMemoryVectorStore.from_documents(
        documents=documents,
        embedding=embeddings
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # Search for relevant documents
    query = "What models does Bedrock support?"
    relevant_docs = retriever.invoke(query)
    
    print(f"Query: {query}")
    print(f"\nRelevant documents found:")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"  {i}. {doc.page_content}")
    
    # Now use LLM to answer based on retrieved docs
    llm = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        region_name="us-east-1"
    )
    
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    response = llm.invoke(f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:""")
    
    print(f"\nAI Answer: {response.content}")


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 LANGCHAIN + BEDROCK EXAMPLES")
    print("=" * 60)
    
    # Run all examples
    basic_langchain_example()
    messages_example()
    streaming_example()
    prompt_template_example()
    conversation_with_memory()
    switch_models_example()
    embeddings_example()
    simple_rag_example()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)