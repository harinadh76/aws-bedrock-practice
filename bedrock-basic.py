"""
LESSON 2: Working with Different Models
========================================
Each model provider has a slightly different request format.
Let's learn how to use the main ones.
"""

import boto3
import json
from dotenv import load_dotenv

load_dotenv()

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DIFFERENT MODEL FORMATS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Each model provider has its own request/response format:                  │
│                                                                              │
│   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐               │
│   │   ANTHROPIC    │  │    AMAZON      │  │     META       │               │
│   │    (Claude)    │  │    (Titan)     │  │    (Llama)     │               │
│   │                │  │                │  │                │               │
│   │  messages: []  │  │  inputText:    │  │   prompt:      │               │
│   │  max_tokens:   │  │  textConfig:   │  │   max_gen_len: │               │
│   └────────────────┘  └────────────────┘  └────────────────┘               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")

# Create client
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')


# ============================================
# Model 1: Anthropic Claude
# ============================================

def call_claude(prompt: str, model_id: str = "anthropic.claude-3-haiku-20240307-v1:0") -> str:
    """
    Call Anthropic Claude models
    
    Available models:
    - anthropic.claude-3-5-sonnet-20240620-v1:0  (Best)
    - anthropic.claude-3-opus-20240229-v1:0      (Most powerful)
    - anthropic.claude-3-sonnet-20240229-v1:0   (Balanced)
    - anthropic.claude-3-haiku-20240307-v1:0    (Fastest)
    """
    
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "temperature": 0.7,  # Creativity (0-1)
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    response = bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps(request_body)
    )
    
    response_body = json.loads(response['body'].read())
    return response_body['content'][0]['text']


# ============================================
# Model 2: Amazon Titan
# ============================================

def call_titan(prompt: str, model_id: str = "amazon.titan-text-express-v1") -> str:
    """
    Call Amazon Titan models
    
    Available models:
    - amazon.titan-text-express-v1   (Fast)
    - amazon.titan-text-lite-v1      (Lightweight)
    - amazon.titan-text-premier-v1:0 (Most capable)
    """
    
    request_body = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 500,
            "temperature": 0.7,
            "topP": 0.9,
            "stopSequences": []
        }
    }
    
    response = bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps(request_body)
    )
    
    response_body = json.loads(response['body'].read())
    return response_body['results'][0]['outputText']


# ============================================
# Model 3: Meta Llama
# ============================================

def call_llama(prompt: str, model_id: str = "meta.llama3-8b-instruct-v1:0") -> str:
    """
    Call Meta Llama models
    
    Available models:
    - meta.llama3-70b-instruct-v1:0  (Largest)
    - meta.llama3-8b-instruct-v1:0   (Faster)
    - meta.llama2-70b-chat-v1        (Previous gen)
    - meta.llama2-13b-chat-v1        (Smaller)
    """
    
    request_body = {
        "prompt": prompt,
        "max_gen_len": 500,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    response = bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps(request_body)
    )
    
    response_body = json.loads(response['body'].read())
    return response_body['generation']


# ============================================
# Model 4: Mistral
# ============================================

def call_mistral(prompt: str, model_id: str = "mistral.mistral-7b-instruct-v0:2") -> str:
    """
    Call Mistral models
    
    Available models:
    - mistral.mistral-large-2402-v1:0    (Most capable)
    - mistral.mixtral-8x7b-instruct-v0:1 (Mixture of experts)
    - mistral.mistral-7b-instruct-v0:2   (Fast)
    """
    
    request_body = {
        "prompt": f"<s>[INST] {prompt} [/INST]",
        "max_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    response = bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps(request_body)
    )
    
    response_body = json.loads(response['body'].read())
    return response_body['outputs'][0]['text']


# ============================================
# Model 5: Cohere
# ============================================

def call_cohere(prompt: str, model_id: str = "cohere.command-text-v14") -> str:
    """
    Call Cohere models
    
    Available models:
    - cohere.command-r-plus-v1:0   (Most capable)
    - cohere.command-r-v1:0        (Balanced)
    - cohere.command-text-v14      (Standard)
    - cohere.command-light-text-v14 (Fast)
    """
    
    request_body = {
        "prompt": prompt,
        "max_tokens": 500,
        "temperature": 0.7,
        "p": 0.9
    }
    
    response = bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps(request_body)
    )
    
    response_body = json.loads(response['body'].read())
    return response_body['generations'][0]['text']


# ============================================
# Universal Function
# ============================================

def call_bedrock(prompt: str, provider: str = "claude") -> str:
    """
    Universal function to call any Bedrock model
    """
    providers = {
        "claude": call_claude,
        "titan": call_titan,
        "llama": call_llama,
        "mistral": call_mistral,
        "cohere": call_cohere
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}")
    
    return providers[provider](prompt)


# ============================================
# Test Different Models
# ============================================

if __name__ == "__main__":
    prompt = "What is the capital of France? Answer in one sentence."
    
    print("Testing different models with the same prompt...")
    print(f"Prompt: {prompt}\n")
    print("=" * 60)
    
    # Test each model (comment out ones you don't have access to)
    models_to_test = [
        ("Claude", "claude"),
        ("Titan", "titan"),
        # ("Llama", "llama"),     # Uncomment if you have access
        # ("Mistral", "mistral"), # Uncomment if you have access
    ]
    
    for model_name, provider in models_to_test:
        try:
            print(f"\n🤖 {model_name}:")
            response = call_bedrock(prompt, provider)
            print(f"   {response.strip()}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)