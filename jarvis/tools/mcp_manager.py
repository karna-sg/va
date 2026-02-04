"""
MCP Client Manager for Jarvis

Manages connections to MCP (Model Context Protocol) servers.
Provides a unified interface for tool calls across all configured servers.

Phase 1: Reads config, registers servers, delegates to Claude CLI.
Phase 2+: Direct MCP calls via Python SDK (bypassing Claude CLI).
"""

import json
import time
from pathlib import Path
from typing import Optional, Any, Dict, List
from dataclasses import dataclass, field


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""
    name: str
    type: str  # 'http', 'stdio'
    url: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class MCPToolResult:
    """Result from an MCP tool call"""
    tool_name: str
    server_name: str
    result: Any
    is_error: bool = False
    error_message: str = ""
    duration_ms: float = 0


class MCPClientManager:
    """
    Manages MCP server connections and tool calls.

    Reads server configurations from .mcp.json and maintains
    a registry of available servers. In Phase 1, all tool calls
    go through Claude CLI. In Phase 2+, Tier 1/2 resolved intents
    will call MCP tools directly.
    """

    def __init__(self, config_path: Optional[str] = None):
        self._servers: Dict[str, MCPServerConfig] = {}
        self._config_path = config_path
        self._connected: Dict[str, bool] = {}

    async def initialize(self, config_path: Optional[str] = None) -> bool:
        """
        Load MCP server configs from .mcp.json.

        Args:
            config_path: Path to .mcp.json file

        Returns:
            True if at least one server was loaded
        """
        path = config_path or self._config_path
        if not path:
            return False

        config_file = Path(path)
        if not config_file.exists():
            print("  MCP config not found: %s" % path)
            return False

        try:
            with open(config_file) as f:
                config = json.load(f)

            servers = config.get('mcpServers', {})
            for name, server_config in servers.items():
                self._servers[name] = MCPServerConfig(
                    name=name,
                    type=server_config.get('type', 'stdio'),
                    url=server_config.get('url'),
                    command=server_config.get('command'),
                    args=server_config.get('args', []),
                    env=server_config.get('env', {}),
                    description=server_config.get('description', ''),
                )
                self._connected[name] = False
                print("  MCP server: %s (%s)" % (name, server_config.get('type', 'stdio')))

            return len(self._servers) > 0

        except Exception as e:
            print("  MCP config error: %s" % e)
            return False

    @property
    def servers(self) -> Dict[str, MCPServerConfig]:
        """Get all registered servers"""
        return self._servers.copy()

    @property
    def server_names(self) -> List[str]:
        """Get list of registered server names"""
        return list(self._servers.keys())

    def is_connected(self, server_name: str) -> bool:
        """Check if a server is connected"""
        return self._connected.get(server_name, False)

    def get_server(self, name: str) -> Optional[MCPServerConfig]:
        """Get server config by name"""
        return self._servers.get(name)

    async def call_tool(self, server_name: str, tool_name: str,
                        arguments: Optional[Dict[str, Any]] = None) -> MCPToolResult:
        """
        Call an MCP tool on a specific server.

        Phase 1: Returns error (all calls go through Claude CLI).
        Phase 2+: Will call MCP servers directly via Python SDK.
        """
        start = time.time()

        server = self._servers.get(server_name)
        if not server:
            return MCPToolResult(
                tool_name=tool_name,
                server_name=server_name,
                result=None,
                is_error=True,
                error_message="Server '%s' not found" % server_name,
            )

        # Phase 1: MCP calls go through Claude CLI
        # Phase 2+: Direct MCP calls via Python SDK will be added here
        return MCPToolResult(
            tool_name=tool_name,
            server_name=server_name,
            result=None,
            is_error=True,
            error_message="Direct MCP calls not yet implemented. Use Claude CLI.",
            duration_ms=(time.time() - start) * 1000,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get status of all MCP servers"""
        return {
            name: {
                'type': server.type,
                'url': server.url,
                'connected': self._connected.get(name, False),
                'description': server.description,
            }
            for name, server in self._servers.items()
        }

    async def shutdown(self) -> None:
        """Shutdown all MCP connections"""
        self._connected = {name: False for name in self._servers}
