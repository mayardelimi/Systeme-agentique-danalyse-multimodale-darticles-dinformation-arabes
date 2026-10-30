# server/hello.py
from mcp.server.mcpserver import MCPServer

mcp = MCPServer('hello')

@mcp.tool()
def hello_world() -> str:
    return "Hello from MCP v2!"

if __name__ == "__main__":
    mcp.run()