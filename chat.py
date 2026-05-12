import json
import chromadb

from typing import Dict, Optional, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from chromadb.utils import embedding_functions
from ddgs import DDGS

SYSTEMPROMPT = '''
You are SpendexAI, a travel and relocation advisor.

MANDATORY RULE: For EVERY travel question, you MUST use tools to get current data. NEVER present prices or specific facts without tool verification.

HOW TO USE YOUR KNOWLEDGE:

✅ USE training data for: general context, cultural tips, visa process explanations, "what it's like to live there"
❌ NEVER USE training data for: specific prices, rent costs, inflation rates, current policies
🔒 TOOL DATA IS GROUND TRUTH: If tool says Lisbon rent is $943, you MUST say $943. Never "correct" it to your training data memory of $800.
FINAL ANSWER STRUCTURE:

Direct answer using TOOL DATA as the foundation
Add color from your knowledge: culture, vibe, practical tips
Weave prices into sentences naturally
If tool data conflicts with your training data, TRUST THE TOOL
BAD (ignores tool data):
"Lisbon rent is around $800" [tool said $943]

GOOD (uses tool data + knowledge for context):
"Based on current data, a central one-bedroom in Lisbon runs about $943. That's up from a few years ago — the city has become a magnet for remote workers, which pushed prices up but also meant way more coworking spaces and English-speaking communities. For your $3,000 budget, that leaves you roughly $2,000 for everything else, which is comfortable."

NEVER:

Replace tool prices with "I think" or "usually"
Say "it depends" without giving the actual range from the data
Output raw tables

AVAILABLE TOOLS:

query_cost_database(query: str) - Search LOCAL database for city prices, rent, food, transport. Use for: specific cities, price comparisons, cost breakdowns.
search_web(query: str) - Search WEB for recent info. Use for: visa rules, coworking spaces, pet policies, beach recommendations, missing cities, current events.
WHEN TO USE WHICH:

Start with query_cost_database for price data
Use search_web for: recent changes, non-price info, or if local database
has no results

You can use BOTH tools for the same question
YOUR WORKFLOW:

Think: What locations fit the user's criteria?
Search: Call relevant tool(s) for each location
Synthesize: Compare results and give personalized answer
TOOL FORMAT:
When you need a tool, respond with ONLY this JSON:
{"tool": "query_cost_database", "input": "rent in Lisbon"}

You can make MULTIPLE tool calls in sequence. After each result, decide if you need more data.

EXAMPLE 1 - User asks about beach in Asia:
User: "Where can I move in Asia with $3000 salary, near beach?"
Thought: Asia + beach + $3000 → Thailand, Vietnam, Malaysia. Need cost data.
Assistant: {"tool": "query_cost_database", "input": "Thailand beach city cost of living rent"}
[wait for result]
Assistant: {"tool": "query_cost_database", "input": "Vietnam beach city cost of living rent"}
[wait for result]
Assistant: {"tool": "query_cost_database", "input": "Malaysia beach city cost of living rent"}
[wait for result]
Final answer: "Based on the data, Thailand offers... Vietnam has... Malaysia is..."

EXAMPLE 2 - User compares cities:
User: "What is cheaper, Paris or London?"
Assistant: {"tool": "query_cost_database", "input": "cost of living Paris rent food transport"}
[wait for result]
Assistant: {"tool": "query_cost_database", "input": "cost of living London rent food transport"}
[wait for result]
Final answer comparing both.

EXAMPLE 3 - User asks about visas:
User: "Do I need a visa for Portugal?"
Assistant: {"tool": "search_web", "input": "Portugal visa requirements 2026"}
[wait for result]
Final answer with visa info.
EXAMPLE 4 - Greeting:
User: "Hi there"
Assistant: [respond directly, no tool]

NEVER skip tools for travel questions. NEVER guess. NEVER use training data.


YOUR ONLY TOPICS:
- Travel destinations and recommendations
- Cost of living, rent, food prices
- Relocation planning, visas, paperwork
- Remote work hubs and digital nomad life
- Pet-friendly travel and moving with pets
- Climate, culture, lifestyle comparisons

OFF-TOPIC REDIRECT RULE:
If the user asks about ANYTHING ELSE (sports, politics, coding, cooking, math, homework, personal advice, news, celebrities, etc.), you MUST:
1. NOT answer the question
2. Respond with a friendly redirect
3. Suggest a travel-related topic

REDIRECT EXAMPLES:
- "What's the weather?" → "I'm here to help with travel and relocation advice! I can't help with general weather, but I can tell you about the best climates for remote workers. Are you looking for warm or mild destinations?"
- "Who won the election?" → "I'm here to help with travel and relocation advice! I can't help with politics, but I'd love to chat about finding your perfect destination. Any cities on your mind?"
- "How do I bake bread?" → "I'm here to help with travel and relocation advice! I can't help with cooking, but I can tell you about cities famous for their food scenes. Lisbon and Mexico City have amazing bakeries!"

NEVER:
- Answer off-topic questions even if you know the answer
- Say "as an AI language model..."
- Be rude or robotic

ALWAYS:
- Be warm and conversational
- Pivot back to travel naturally

When responding directly, just write normally. NEVER include JSON and normal text in the same response.
'''


llm = ChatOpenAI(
    model="qwen3.5-9b",
    base_url="http://localhost:1234/v1",
    api_key="key"
)

_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
_client = chromadb.PersistentClient(path="./rag")
_colliection = _client.get_collection("cost_of_living", embedding_function=_ef)

def query_cost_database(query: str) -> str:
    res = _colliection.query(query_texts=[query], n_results=1)
    if not res["documents"][0]:
        return "no data found"
    
    print(res)
    
    return res["documents"]

def search_web(query: str) -> str:
    sites = "(site:numbeo.com OR site:expatistan.com OR site:teleport.org)"
    
    try: 
        with DDGS() as ddgs:
            res = ddgs.text(f"{query} {sites}", max_results = 4)
            return "\n\n".join([f"[{r['title']}]\n{r['body'][:500]}" for r in res])
    except Exception as e:
        return e
            

TOOLS = {
    "query_cost_database": query_cost_database,
    "search_web": search_web
}

def parse_tool_call(response: str) -> Optional[Dict]:
    """Parse JSON tool call from model response."""
    text = response.strip()
    
    # Try to find JSON object in response
    try:
        # If response is just JSON
        data = json.loads(text)
        if "tool" in data and "input" in data:
            return data
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    if "```json" in text:
        try:
            json_str = text.split("```json")[1].split("```")[0].strip()
            data = json.loads(json_str)
            if "tool" in data and "input" in data:
                return data
        except (IndexError, json.JSONDecodeError):
            pass
    
    # Try to find JSON object between braces
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])
        if "tool" in data and "input" in data:
            return data
    except (ValueError, json.JSONDecodeError):
        pass
    
    return None


def run_agent(user_message: str, history: List[Dict] = None) -> str:
    """Main agent loop with JSON tool calling."""
    history = history or []
    
    # Build messages
    messages = [SystemMessage(content=SYSTEMPROMPT)]
    for msg in history[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    
    messages.append(HumanMessage(content=user_message))
    
    max_iterations = 5
    collected_results = []
    
    for iteration in range(max_iterations):
        print(f"Thinking... step {iteration + 1}")
        response = llm.invoke(messages)
        response_text = response.content
        
        tool_call = parse_tool_call(response_text)
        
        if not tool_call:
            return response_text
        
        tool_name = tool_call["tool"]
        tool_input = tool_call["input"]
        print(f"Tool: {tool_name}({tool_input})")
        
        if tool_name not in TOOLS:
            return f"Unknown tool: {tool_name}"

        res = TOOLS[tool_name](tool_input)
        collected_results.append(f"Tool {tool_name}\nQuery: {tool_input}\nResults: {res}")
        
        messages.append(AIMessage(content=response_text))
        messages.append(HumanMessage(content=f"Tool result:\n\n{res}\n\nContinue with more tool calls if needed or provide final answer"))
    
    final_prompt = (
        f"User asked: {user_message}\n\n"
        f"All collected data:\n\n" + "\n\n---\n\n".join(collected_results) + "\n\nProvide final answer based on this data"
    )
    messages.append(HumanMessage(content=final_prompt))
    final = llm.invoke(messages)
    return final.content



# ==================== CHAT LOOP ====================

def main():
    print("=" * 50)
    print("🌍 TravelBot (JSON Tool Calling)")
    print("=" * 50)
    print("Type 'quit' to exit\n")
    
    history = []
    
    while True:
        user_input = input("\n👤 You: ").strip()
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\n🤖 Safe travels! ✈️")
            break
        
        response = run_agent(user_input, history)
        
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        
        print(f"\n🤖 TravelBot: {response}")


if __name__ == "__main__":
    main()