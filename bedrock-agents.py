"""
LESSON 7: Bedrock Agents
=========================
Create AI agents that can take actions and use tools
"""

import boto3
import json
from dotenv import load_dotenv

load_dotenv()

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BEDROCK AGENTS                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Agents = LLM + Tools + Decision Making                                    │
│                                                                              │
│   Regular LLM:                                                              │
│   ┌──────┐         ┌───────┐         ┌──────────┐                          │
│   │ User │────────▶│  LLM  │────────▶│  Answer  │                          │
│   └──────┘         └───────┘         └──────────┘                          │
│   (Can only generate text - can't DO anything)                              │
│                                                                              │
│   Bedrock Agent:                                                            │
│   ┌──────┐         ┌───────────────────────────────────┐                   │
│   │ User │────────▶│           AGENT                   │                   │
│   └──────┘         │   ┌─────────────────────────┐    │                   │
│                    │   │         LLM              │    │                   │
│                    │   │    (Brain/Reasoning)     │    │                   │
│                    │   └───────────┬─────────────┘    │                   │
│                    │               │ Can decide to... │                   │
│                    │   ┌───────────┼───────────────┐  │                   │
│                    │   │           │               │  │                   │
│                    │   ▼           ▼               ▼  │                   │
│                    │ ┌─────┐   ┌─────────┐   ┌──────┐ │                   │
│                    │ │ API │   │Database │   │Search│ │                   │
│                    │ │Call │   │  Query  │   │ Web  │ │                   │
│                    │ └─────┘   └─────────┘   └──────┘ │                   │
│                    │      (Action Groups/Tools)       │                   │
│                    └───────────────────────────────────┘                   │
│                                    │                                        │
│                                    ▼                                        │
│                           ┌──────────────┐                                 │
│                           │ Task Complete│                                 │
│                           │   + Answer   │                                 │
│                           └──────────────┘                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")


# ============================================
# Invoking an Existing Bedrock Agent
# ============================================

class BedrockAgent:
    """
    Client for invoking Bedrock Agents
    
    Note: Agents must be created in AWS Console first.
    This class shows how to INVOKE an existing Agent.
    """
    
    def __init__(self, agent_id: str, agent_alias_id: str, region: str = "us-east-1"):
        self.agent_id = agent_id
        self.agent_alias_id = agent_alias_id
        self.client = boto3.client('bedrock-agent-runtime', region_name=region)
    
    def invoke(self, prompt: str, session_id: str = None) -> dict:
        """
        Invoke the agent with a prompt
        """
        import uuid
        
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        response = self.client.invoke_agent(
            agentId=self.agent_id,
            agentAliasId=self.agent_alias_id,
            sessionId=session_id,
            inputText=prompt
        )
        
        # Parse the streaming response
        result = ""
        for event in response['completion']:
            if 'chunk' in event:
                chunk = event['chunk']
                result += chunk.get('bytes', b'').decode('utf-8')
        
        return {
            'response': result,
            'session_id': session_id
        }


# ============================================
# Creating an Agent (Console Steps)
# ============================================

def print_agent_creation_steps():
    """
    Print the steps to create an Agent in AWS Console
    """
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│           CREATING A BEDROCK AGENT (AWS Console)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   STEP 1: Create Agent                                                      │
│   ─────────────────────                                                     │
│   1. Go to Amazon Bedrock Console                                           │
│   2. Click "Agents" in the left sidebar                                     │
│   3. Click "Create Agent"                                                   │
│   4. Enter agent name: "customer-service-agent"                            │
│   5. Enter description                                                       │
│   6. Select a foundation model (e.g., Claude 3 Sonnet)                      │
│                                                                              │
│   STEP 2: Configure Instructions                                            │
│   ──────────────────────────────                                            │
│   Write instructions telling the agent its role:                            │
│                                                                              │
│   "You are a customer service agent for TechCo. You can:                    │
│    - Look up order status                                                   │
│    - Process returns                                                        │
│    - Answer product questions                                               │
│    Be helpful and professional."                                            │
│                                                                              │
│   STEP 3: Add Action Groups                                                 │
│   ─────────────────────────                                                 │
│   Action Groups define what the agent can DO:                               │
│                                                                              │
│   Option A: Lambda Function                                                 │
│   ┌─────────────────────────────────────────────────────┐                  │
│   │  def lambda_handler(event, context):                │                  │
│   │      action = event['actionGroup']                  │                  │
│   │      function = event['function']                   │                  │
│   │      parameters = event['parameters']               │                  │
│   │                                                     │                  │
│   │      if function == 'get_order_status':            │                  │
│   │          order_id = parameters[0]['value']         │                  │
│   │          # Look up order in database               │                  │
│   │          return {"status": "shipped"}              │                  │
│   └─────────────────────────────────────────────────────┘                  │
│                                                                              │
│   Option B: OpenAPI Schema (for existing APIs)                              │
│   ┌─────────────────────────────────────────────────────┐                  │
│   │  openapi: 3.0.0                                     │                  │
│   │  paths:                                             │                  │
│   │    /orders/{orderId}:                               │                  │
│   │      get:                                           │                  │
│   │        summary: Get order status                    │                  │
│   │        parameters:                                  │                  │
│   │          - name: orderId                            │                  │
│   │            in: path                                 │                  │
│   │            required: true                           │                  │
│   └─────────────────────────────────────────────────────┘                  │
│                                                                              │
│   STEP 4: (Optional) Add Knowledge Base                                     │
│   ──────────────────────────────────────                                    │
│   Connect a Knowledge Base for document Q&A                                 │
│                                                                              │
│   STEP 5: Create & Test                                                     │
│   ──────────────────────                                                    │
│   1. Click "Create Agent"                                                   │
│   2. Click "Prepare" to prepare the agent                                   │
│   3. Use the test console to try it out                                     │
│   4. Create an Alias for production use                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")


# ============================================
# Building Agents with LangGraph (Alternative)
# ============================================

from langchain_aws import ChatBedrock
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

def langgraph_agent_example():
    """
    Build an agent using LangGraph instead of Bedrock Agents
    This gives you more control and doesn't require AWS Console setup
    """
    
    print("\n📘 LangGraph Agent Example")
    print("=" * 50)
    
    # Define tools the agent can use
    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        # In real app, call a weather API
        weather_data = {
            "new york": "72°F, Sunny",
            "london": "58°F, Cloudy",
            "tokyo": "68°F, Partly Cloudy",
        }
        return weather_data.get(city.lower(), f"Weather data not available for {city}")
    
    @tool
    def calculate(expression: str) -> str:
        """Calculate a mathematical expression. Example: '2 + 2' or '10 * 5'"""
        try:
            # Simple eval for demo - in production use a safe parser
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error calculating: {e}"
    
    @tool
    def search_products(query: str) -> str:
        """Search for products in the catalog."""
        # Mock product database
        products = {
            "laptop": [
                {"name": "ProBook 15", "price": 999},
                {"name": "UltraLight X1", "price": 1299},
            ],
            "phone": [
                {"name": "SmartPhone Pro", "price": 799},
                {"name": "BudgetPhone", "price": 299},
            ],
        }
        
        results = []
        for key, items in products.items():
            if query.lower() in key:
                results.extend(items)
        
        if results:
            return json.dumps(results)
        return "No products found matching your query."
    
    # Create the LLM
    llm = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        region_name="us-east-1"
    )
    
    # Create the agent
    tools = [get_weather, calculate, search_products]
    agent = create_react_agent(llm, tools)
    
    # Test the agent
    test_queries = [
        "What's the weather in Tokyo?",
        "Calculate 15 * 7 + 23",
        "Search for laptops",
        "What's 100 divided by 4, and also what's the weather in London?",
    ]
    
    for query in test_queries:
        print(f"\n❓ User: {query}")
        
        result = agent.invoke({"messages": [("human", query)]})
        
        # Get the final response
        final_message = result["messages"][-1]
        print(f"🤖 Agent: {final_message.content}")


# ============================================
# Custom Tool-Calling Agent (Manual Implementation)
# ============================================

def manual_tool_calling_example():
    """
    Manually implement tool calling to understand how it works
    """
    
    print("\n📘 Manual Tool Calling Example")
    print("=" * 50)
    
    # Tools available to the agent
    def get_order_status(order_id: str) -> dict:
        """Simulate looking up order status"""
        orders = {
            "ORD-123": {"status": "Shipped", "eta": "Dec 25"},
            "ORD-456": {"status": "Processing", "eta": "Dec 28"},
            "ORD-789": {"status": "Delivered", "delivered_date": "Dec 20"},
        }
        return orders.get(order_id, {"error": "Order not found"})
    
    def process_refund(order_id: str, reason: str) -> dict:
        """Simulate processing a refund"""
        return {
            "success": True,
            "message": f"Refund initiated for {order_id}",
            "reason": reason,
            "refund_id": "REF-001"
        }
    
    tools_description = """
    Available tools:
    1. get_order_status(order_id: str) - Get the status of an order
    2. process_refund(order_id: str, reason: str) - Process a refund request
    
    To use a tool, respond with JSON: {"tool": "tool_name", "params": {...}}
    After getting tool results, provide a final answer to the user.
    """
    
    llm = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    def call_llm(messages: list) -> str:
        response = llm.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "system": f"""You are a customer service agent. {tools_description}""",
                "messages": messages
            })
        )
        return json.loads(response['body'].read())['content'][0]['text']
    
    def run_agent(user_query: str, max_iterations: int = 5) -> str:
        """Run the agent loop"""
        messages = [{"role": "user", "content": user_query}]
        
        for i in range(max_iterations):
            response = call_llm(messages)
            
            # Check if response contains a tool call
            try:
                if '{"tool"' in response:
                    # Extract JSON from response
                    start = response.find('{')
                    end = response.rfind('}') + 1
                    tool_call = json.loads(response[start:end])
                    
                    # Execute the tool
                    tool_name = tool_call['tool']
                    params = tool_call.get('params', {})
                    
                    print(f"   🔧 Calling tool: {tool_name}({params})")
                    
                    if tool_name == "get_order_status":
                        result = get_order_status(params.get('order_id'))
                    elif tool_name == "process_refund":
                        result = process_refund(
                            params.get('order_id'),
                            params.get('reason', '')
                        )
                    else:
                        result = {"error": f"Unknown tool: {tool_name}"}
                    
                    # Add tool result to conversation
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user", 
                        "content": f"Tool result: {json.dumps(result)}"
                    })
                else:
                    # No tool call - this is the final answer
                    return response
            except json.JSONDecodeError:
                # Response is not a tool call - return as final answer
                return response
        
        return "Max iterations reached"
    
    # Test
    print("\nTest 1: Order Status Query")
    result = run_agent("What's the status of order ORD-123?")
    print(f"🤖 Agent: {result}")
    
    print("\nTest 2: Refund Request")
    result = run_agent("I want to return order ORD-456 because it was damaged")
    print(f"🤖 Agent: {result}")


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 BEDROCK AGENTS")
    print("=" * 60)
    
    # Print creation steps
    print_agent_creation_steps()
    
    # Run LangGraph agent example
    langgraph_agent_example()
    
    # Run manual tool calling example
    manual_tool_calling_example()