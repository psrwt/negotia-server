from typing import List
from langchain_core.tools import tool
from config import embeddings_model, llm
from vector_store import index

@tool
def find_product(query: str) -> List[dict]:
    """Gets product from Pinecone."""
    print(f"--- TOOL: find_product(query='{query}') ---")
    query_embedding = embeddings_model.embed_query(query)
    results = index.query(vector=query_embedding, top_k=7, include_metadata=True, namespace="mobiles")
    products = []
    for match in results['matches']:
        metadata = match.get('metadata', {})
        product_data = {
            "ID": match.get('id'),
            "Company Name": metadata.get("Company Name"),
            "Model Name": metadata.get("Model Name"),
            "Max Price": metadata.get("Max Price"),
            "Capacity": metadata.get("Capacity"),
            "ram": metadata.get("ram"),
            "Back Camera": metadata.get("Back Camera"),
            "Front Camera": metadata.get("Front Camera"),
            "Processor": metadata.get("Processor"),
            "Screen Size": metadata.get("Screen Size"),
            "battery": metadata.get("battery"),
            "Text": metadata.get("Text"),
            "image_url": metadata.get("image_url")
        }
        if all(k in product_data and product_data[k] is not None for k in ["ID", "Model Name", "Max Price"]):
            products.append(product_data)
    return products

# @tool
# def get_deal(conversation_context: str, product_ids: List[str]) -> str:
#     """
#     Analyzes the conversation and a list of relevant products to generate a special deal,
#     cross-sell, or upsell offer. Use this when the user shows buying intent, asks for discounts,
#     or compares products.
#     """
#     print(f"--- TOOL: get_deal(context='{conversation_context}', product_ids={product_ids}) ---")
#     deal_prompt = f"""
#     Based on the conversation context: '{conversation_context}', create a compelling, short, one-sentence deal heading
#     and a deal price for one of the products from this list: {product_ids}.
#     Format the output as a simple JSON string with keys 'heading' and 'deal_price'.
#     Example: {{"heading": "Limited Time: 15% off the Pixel 8 Pro!", "deal_price": 59415}}
#     """
#     response = llm.invoke(deal_prompt)
#     return response.content

@tool
def get_deal(conversation_context: str, product_ids: List[str]) -> str:
    """
    Disabled. The agent should negotiate directly through conversation.
    """
    print("--- TOOL: get_deal CALLED BUT DISABLED ---")
    return "NOTE: Deal tool disabled. Negotiate naturally."