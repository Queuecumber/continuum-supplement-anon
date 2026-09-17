"""Export the actual MCP tool definitions, without loading inference models."""
import argparse
import asyncio
import json
from pathlib import Path


async def export(output):
    from continuum.server.standard import mcp

    tools = await mcp.list_tools()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([tool.model_dump(mode='json', exclude_none=True)
                                  for tool in tools], indent=2, ensure_ascii=True) + '\n')
    print(f'Exported {len(tools)} MCP tool definitions to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    asyncio.run(export(parser.parse_args().out))
