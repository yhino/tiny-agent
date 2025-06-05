from datetime import datetime
from zoneinfo import ZoneInfo

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("clock")


@mcp.tool()
def current_time(timezone: str = "Asia/Tokyo") -> str:
    """Returns the current time in ISO 8601 format"""
    tz = ZoneInfo(timezone)
    return datetime.now().astimezone(tz).isoformat()


if __name__ == "__main__":
    mcp.run(transport="stdio")
