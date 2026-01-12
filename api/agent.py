import json
import operator
import re
from typing import TypedDict, Annotated, Dict, List
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from schemas import Product, FinalAnswer
from tools import find_product, get_deal
from config import llm

# --- Agent Setup ---
tools = [find_product, get_deal]
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    retrieved_products: Annotated[Dict[str, Product], operator.ior]
    product_context_ids: list[str]
    detected_language: str  # Track the detected language

def detect_language_from_text(text: str) -> str:
    """Detect language from text using character patterns."""
    language_patterns = {
        'hi-IN': r'[\u0900-\u097F]',  # Hindi
        'ko-KR': r'[\uAC00-\uD7AF]',  # Korean
        'ja-JP': r'[\u3040-\u309F\u30A0-\u30FF]',  # Japanese
        'zh-CN': r'[\u4E00-\u9FFF]',  # Chinese
        'es-ES': r'[áéíóúñ¿¡]',  # Spanish
        'fr-FR': r'[àâçéèêëîïôûùüÿœæ]',  # French
        'de-DE': r'[äöüß]',  # German
        'it-IT': r'[àèéìíîòóùú]',  # Italian
        'pt-BR': r'[ãõâêîôûáéíóúç]',  # Portuguese
        'ar-SA': r'[\u0600-\u06FF]',  # Arabic
    }
    
    for lang_code, pattern in language_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return lang_code
    
    return 'en-US'  # Default to English

def tool_using_agent_node(state: AgentState):
    """Decides to call a tool or respond."""
    print("--- NODE: Tool Using Agent ---")
    
    # Detect language from the latest user message
    detected_language = state.get("detected_language", "en-US")
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            detected_language = detect_language_from_text(msg.content)
            break
    
    product_context = state.get("product_context_ids", [])
    
    # VERY STRONG multilingual instruction
    # system_prompt = f"""
    # CRITICAL LANGUAGE INSTRUCTION: YOU MUST RESPOND IN {detected_language.upper()}!
    
    # You are an expert multilingual mobile sales assistant. The user is speaking in {detected_language}.
    
    # ABSOLUTE RULES:
    # 1. RESPOND ONLY IN {detected_language.upper()} - NEVER IN ENGLISH OR ANY OTHER LANGUAGE
    # 2. If user says "사랑해" (Korean), respond in Korean with appropriate mobile sales response
    # 3. If user says Hindi text, respond in Hindi
    # 4. If user says any non-English text, respond in that same language
    # 5. NEVER say "I only speak English" or similar - you are fully multilingual
    
    # BUSINESS RULES:
    # - You sell mobile phones and accessories
    # - Current Product Context: {product_context if product_context else "None"}
    # - Use `find_product` for new product searches
    # - If the user shows buying intent or asks for discounts, negotiate naturally in conversation.
    #     Do NOT call any tool for pricing, discounts, or deals.
    #     Respond conversationally with logical negotiation steps.
    #     Avoid returning JSON. Avoid using the get_deal tool.
    
    # EXAMPLE RESPONSES:
    # - Korean input: "사랑해" → Korean response about mobile offers
    # - Hindi input: "आप हिंदी में बात कर सकते हैं" → Hindi response confirming multilingual capability
    # - English input: "Hello" → English response
    # """

    system_prompt = f"""
        CRITICAL LANGUAGE INSTRUCTION: YOU MUST RESPOND IN {detected_language.upper()}!
        
        You are an expert multilingual mobile sales assistant. The user is speaking in {detected_language}.
        
        ABSOLUTE RULES:
        1. RESPOND ONLY IN {detected_language.upper()} - NEVER IN ENGLISH OR ANY OTHER LANGUAGE
        2. If user says "사랑해" (Korean), respond in Korean with appropriate mobile sales response
        3. If user says Hindi text, respond in Hindi
        4. If user says any non-English text, respond in that same language
        5. NEVER say "I only speak English" or similar - you are fully multilingual
        
        BUSINESS RULES:
        - You sell mobile phones and accessories
        - Current Product Context: {product_context if product_context else "None"}
        - Use `find_product` for new product searches

        PRODUCT PRICE REFERENCE TABLE:
        1. Pixel 8 (Google - 128GB) → ₹59,900
        2. iPhone 15 Pro (Apple - 256GB) → ₹89,900
        3. Galaxy S24 Ultra (Samsung - 256GB) → ₹1,14,900
        4. Pixel 8 Pro (Google - 256GB) → ₹69,900
        5. iPhone 15 Pro (Apple - 128GB) → ₹79,900
        6. Galaxy S24 Ultra (Samsung - 128GB) → ₹1,04,900

        NEGOTIATION LOGIC:
        - If the user shows buying intent or asks for discounts, negotiate naturally in conversation.
        - Do NOT call any tool for pricing, discounts, or deals.
        - Do NOT return JSON.
        - Respond conversationally like a real sales agent.
        - Start with small discounts, give slightly better deals if user insists, but never unrealistic offers.

        EXAMPLE RESPONSES:
        - Korean input: "사랑해" → Korean response about mobile offers
        - Hindi input: "आप हिंदी में बात कर सकते हैं" → Hindi response confirming multilingual capability
        - English input: "Hello" → English response
    """

    
    # Prepare messages for LLM (keep only non-system messages)
    filtered_messages = []
    for msg in state["messages"]:
        if not isinstance(msg, SystemMessage) and not (hasattr(msg, 'type') and getattr(msg, 'type', None) == 'system'):
            filtered_messages.append(msg)
    
    response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + filtered_messages)
    return {"messages": [response], "detected_language": detected_language}

def tool_node(state: AgentState):
    """Executes tools."""
    print("--- NODE: Tool Executor ---")
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []
    retrieved_products_update = {}
    current_context_ids = state.get("product_context_ids", [])

    for tool_call in tool_calls:
        tool_output = globals()[tool_call['name']].invoke(tool_call['args'])
        if tool_call['name'] == 'find_product':
            current_context_ids = [p['ID'] for p in tool_output]
            for product_dict in tool_output:
                product_obj = Product(**product_dict)
                retrieved_products_update[product_obj.id] = product_obj
            output_str = json.dumps(tool_output)
        else:
            output_str = str(tool_output)
        tool_messages.append(ToolMessage(content=output_str, tool_call_id=tool_call['id']))

    return {
        "messages": tool_messages,
        "retrieved_products": retrieved_products_update,
        "product_context_ids": current_context_ids
    }

def final_answer_node(state: AgentState):
    """Formats the final response."""
    print("--- NODE: Final Answer Formatter ---")
    formatter_llm = llm.with_structured_output(FinalAnswer)
    
    detected_language = state.get("detected_language", "en-US")
    
    # Strong formatting instruction with language enforcement
    formatting_prompt = f"""
    FORMAT THE FINAL RESPONSE USING THE `FinalAnswer` TOOL.
    
    CRITICAL: YOU MUST RESPOND IN {detected_language.upper()}!
    
    RULES:
    1. Text must be in {detected_language}
    2. Maintain conversational tone appropriate for {detected_language}
    3. Include relevant product_ids from find_product tool output
    4. Never translate to English - keep everything in {detected_language}
    
    Examples:
    - Korean: Use Korean characters and grammar
    - Hindi: Use Devanagari script
    - Japanese: Use appropriate Japanese characters
    """
    
    # Filter out system messages for the formatting call
    filtered_messages = []
    for msg in state["messages"]:
        if not isinstance(msg, SystemMessage) and not (hasattr(msg, 'type') and getattr(msg, 'type', None) == 'system'):
            filtered_messages.append(msg)
    
    response = formatter_llm.invoke([SystemMessage(content=formatting_prompt)] + filtered_messages)
    return {"messages": [AIMessage(content="", tool_calls=[{"name": "FinalAnswer", "args": response.model_dump(), "id": "final"}])]}

def router(state: AgentState) -> str:
    """Decides the next step."""
    print("--- ROUTER ---")
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "final_answer_formatter"

# --- Graph Definition ---
workflow = StateGraph(AgentState)
workflow.add_node("agent", tool_using_agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("final_answer_formatter", final_answer_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", router)
workflow.add_edge("tools", "agent")
workflow.add_edge("final_answer_formatter", END)
chatbot_graph = workflow.compile()