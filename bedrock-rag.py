"""
LESSON 6: Bedrock Knowledge Bases
==================================
Build a RAG system using Bedrock's managed Knowledge Bases
"""

import boto3
import json
from dotenv import load_dotenv

load_dotenv()

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BEDROCK KNOWLEDGE BASES                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Knowledge Bases = Your documents + Vector search + LLM                    │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                      │   │
│   │   1. INGEST                                                         │   │
│   │   ┌──────────┐     ┌──────────────┐     ┌────────────────┐         │   │
│   │   │  Your    │────▶│   Chunking   │────▶│   Embeddings   │         │   │
│   │   │  Docs    │     │   & Processing│     │   (Titan)      │         │   │
│   │   │ (S3)     │     │              │     │                │         │   │
│   │   └──────────┘     └──────────────┘     └───────┬────────┘         │   │
│   │                                                  │                   │   │
│   │                                                  ▼                   │   │
│   │                                          ┌────────────────┐         │   │
│   │                                          │  Vector Store  │         │   │
│   │                                          │  (OpenSearch)  │         │   │
│   │                                          └────────────────┘         │   │
│   │                                                  │                   │   │
│   │   2. QUERY                                       │                   │   │
│   │   ┌──────────┐     ┌──────────────┐             │                   │   │
│   │   │  User    │────▶│   Semantic   │◀────────────┘                   │   │
│   │   │  Query   │     │   Search     │                                 │   │
│   │   └──────────┘     └──────┬───────┘                                 │   │
│   │                           │                                          │   │
│   │                           ▼                                          │   │
│   │   ┌──────────────────────────────────────────────────────────┐     │   │
│   │   │   LLM generates answer using retrieved context           │     │   │
│   │   └──────────────────────────────────────────────────────────┘     │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")


# ============================================
# Knowledge Base Client
# ============================================

class BedrockKnowledgeBase:
    """
    Client for working with Bedrock Knowledge Bases
    
    Note: Knowledge Bases must be created in AWS Console first.
    This class shows how to QUERY an existing Knowledge Base.
    """
    
    def __init__(self, knowledge_base_id: str, region: str = "us-east-1"):
        self.knowledge_base_id = knowledge_base_id
        self.client = boto3.client('bedrock-agent-runtime', region_name=region)
    
    def retrieve(self, query: str, num_results: int = 5) -> list:
        """
        Retrieve relevant documents from the Knowledge Base
        """
        response = self.client.retrieve(
            knowledgeBaseId=self.knowledge_base_id,
            retrievalQuery={
                'text': query
            },
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': num_results
                }
            }
        )
        
        results = []
        for item in response.get('retrievalResults', []):
            results.append({
                'content': item.get('content', {}).get('text', ''),
                'score': item.get('score', 0),
                'source': item.get('location', {}).get('s3Location', {}).get('uri', 'Unknown')
            })
        
        return results
    
    def retrieve_and_generate(
        self, 
        query: str, 
        model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    ) -> dict:
        """
        Retrieve documents and generate an answer using an LLM
        """
        response = self.client.retrieve_and_generate(
            input={
                'text': query
            },
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': self.knowledge_base_id,
                    'modelArn': f'arn:aws:bedrock:us-east-1::foundation-model/{model_id}'
                }
            }
        )
        
        return {
            'answer': response.get('output', {}).get('text', ''),
            'citations': response.get('citations', [])
        }


# ============================================
# Creating a Knowledge Base (Console Steps)
# ============================================

def print_kb_creation_steps():
    """
    Print the steps to create a Knowledge Base in AWS Console
    """
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│           CREATING A KNOWLEDGE BASE (AWS Console)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   STEP 1: Prepare Your Documents                                            │
│   ────────────────────────────────                                          │
│   • Create an S3 bucket                                                     │
│   • Upload your documents (PDF, TXT, MD, HTML, DOC, CSV)                   │
│   • Example: s3://my-kb-bucket/documents/                                   │
│                                                                              │
│   STEP 2: Create Knowledge Base                                             │
│   ─────────────────────────────                                             │
│   1. Go to Amazon Bedrock Console                                           │
│   2. Click "Knowledge bases" in the left sidebar                           │
│   3. Click "Create knowledge base"                                          │
│   4. Enter a name: "my-company-kb"                                          │
│   5. Create or select an IAM role                                           │
│                                                                              │
│   STEP 3: Configure Data Source                                             │
│   ────────────────────────────                                              │
│   1. Select "Amazon S3" as data source                                      │
│   2. Enter your S3 bucket URI                                               │
│   3. Choose chunking strategy:                                              │
│      • Fixed size (recommended to start)                                    │
│      • Default chunking                                                     │
│      • No chunking                                                          │
│                                                                              │
│   STEP 4: Select Embeddings Model                                           │
│   ────────────────────────────────                                          │
│   • Titan Embeddings V2 (recommended)                                       │
│   • Cohere Embed                                                            │
│                                                                              │
│   STEP 5: Configure Vector Store                                            │
│   ─────────────────────────────                                             │
│   • Quick create (Amazon OpenSearch Serverless) - easiest                  │
│   • Or use existing OpenSearch/Pinecone/Redis                              │
│                                                                              │
│   STEP 6: Create & Sync                                                     │
│   ─────────────────────                                                     │
│   1. Click "Create knowledge base"                                          │
│   2. Wait for creation (few minutes)                                        │
│   3. Click "Sync" to index your documents                                   │
│                                                                              │
│   STEP 7: Get Knowledge Base ID                                             │
│   ────────────────────────────                                              │
│   • Copy the Knowledge Base ID (looks like: XXXXXXXXXX)                    │
│   • You'll use this in your code                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")


# ============================================
# Example Usage (with existing KB)
# ============================================

def example_query_kb():
    """
    Example of querying an existing Knowledge Base
    """
    
    # Replace with your actual Knowledge Base ID
    KNOWLEDGE_BASE_ID = "YOUR_KB_ID_HERE"
    
    if KNOWLEDGE_BASE_ID == "YOUR_KB_ID_HERE":
        print("""
⚠️  To run this example:
   1. Create a Knowledge Base in AWS Console (see steps above)
   2. Replace 'YOUR_KB_ID_HERE' with your actual Knowledge Base ID
   3. Run this script again
        """)
        return
    
    kb = BedrockKnowledgeBase(KNOWLEDGE_BASE_ID)
    
    # Example 1: Just retrieve documents
    print("📚 Retrieving relevant documents...")
    query = "What is the refund policy?"
    
    results = kb.retrieve(query, num_results=3)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} relevant documents:\n")
    
    for i, result in enumerate(results, 1):
        print(f"  {i}. Score: {result['score']:.3f}")
        print(f"     Content: {result['content'][:200]}...")
        print(f"     Source: {result['source']}\n")
    
    # Example 2: Retrieve and generate answer
    print("\n💬 Generating answer with context...")
    
    response = kb.retrieve_and_generate(query)
    
    print(f"Answer: {response['answer']}")
    
    if response['citations']:
        print("\nSources:")
        for citation in response['citations']:
            print(f"  • {citation}")


# ============================================
# Local RAG Alternative (No KB Required)
# ============================================

from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def local_rag_example():
    """
    Build a RAG system locally without using Bedrock Knowledge Bases
    Good for learning and small-scale applications
    """
    
    print("\n📘 Local RAG Example (No AWS KB Required)")
    print("=" * 50)
    
    # Sample documents (in real app, load from files)
    documents = [
        Document(
            page_content="""
            Our return policy allows customers to return products within 30 days 
            of purchase. Items must be unused and in original packaging. 
            Refunds are processed within 5-7 business days.
            """,
            metadata={"source": "policy.pdf", "page": 1}
        ),
        Document(
            page_content="""
            Shipping is free for orders over $50. Standard shipping takes 5-7 
            business days. Express shipping (2-3 days) is available for $9.99.
            International shipping rates vary by destination.
            """,
            metadata={"source": "policy.pdf", "page": 2}
        ),
        Document(
            page_content="""
            Our customer support is available 24/7 via chat and email. 
            Phone support is available Monday-Friday 9AM-6PM EST.
            For urgent issues, use our priority support line.
            """,
            metadata={"source": "support.pdf", "page": 1}
        ),
        Document(
            page_content="""
            Product warranty covers manufacturing defects for 1 year from 
            purchase date. Extended warranty options are available at checkout.
            Warranty does not cover damage from misuse.
            """,
            metadata={"source": "warranty.pdf", "page": 1}
        ),
    ]
    
    # Create embeddings
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        region_name="us-east-1"
    )
    
    # Create vector store
    vectorstore = InMemoryVectorStore.from_documents(
        documents=documents,
        embedding=embeddings
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # Create LLM
    llm = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        region_name="us-east-1"
    )
    
    # Create RAG prompt
    rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful customer service assistant. Answer the question based ONLY 
on the following context. If the context doesn't contain the answer, say 
"I don't have information about that in my knowledge base."

Context:
{context}

Question: {question}

Answer:""")
    
    # Create RAG chain
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    # Test queries
    test_queries = [
        "What is the return policy?",
        "How long does shipping take?",
        "Is there a warranty?",
        "What are your business hours for phone support?",
    ]
    
    for query in test_queries:
        print(f"\n❓ Question: {query}")
        answer = rag_chain.invoke(query)
        print(f"💬 Answer: {answer}")


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 BEDROCK KNOWLEDGE BASES")
    print("=" * 60)
    
    # Print creation steps
    print_kb_creation_steps()
    
    # Try the example query (won't work without a real KB ID)
    example_query_kb()
    
    # Run local RAG example (works without KB)
    local_rag_example()