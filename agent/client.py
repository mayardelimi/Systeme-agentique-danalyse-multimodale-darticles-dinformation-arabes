import asyncio, json, time, sys, yaml
from litellm import acompletion
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

cfg = yaml.safe_load(open("agent/config.yaml"))

SERVER = StdioServerParameters(command=sys.executable, args=["server/server.py"])

def to_llm_tools(mcp_tools):
    return [{"type": "function", "function": {
        "name": t.name, "description": t.description,
        "parameters": t.inputSchema}} for t in mcp_tools]

async def run(user_prompt: str):
    async with stdio_client(SERVER) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = to_llm_tools((await session.list_tools()).tools)
            messages = [{"role": "user", "content": user_prompt}]

            for _ in range(cfg["max_steps"]):
                resp = await acompletion(model=cfg["model"], messages=messages,
                                         tools=tools, temperature=0)
                msg = resp.choices[0].message
                messages.append(msg)
                if not msg.tool_calls:
                    return msg.content
                for call in msg.tool_calls:
                    args = json.loads(call.function.arguments or "{}")
                    t0 = time.time()
                    result = await session.call_tool(call.function.name, args)
                    text = result.content[0].text
                    with open("eval/trace.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps({"tool": call.function.name, "args": args,
                                            "ms": round((time.time()-t0)*1000)},
                                           ensure_ascii=False) + "\n")
                    messages.append({"role": "tool", "tool_call_id": call.id,
                                     "content": text})
    return "max steps reached"

if __name__ == "__main__":
    print(asyncio.run(run(sys.argv[1])))