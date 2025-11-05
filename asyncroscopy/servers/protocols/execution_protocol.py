from twisted.protocols.basic import Int32StringReceiver
import traceback


class ExecutionProtocol(Int32StringReceiver):
    """
    Executes locally registered commands.
    Used by backend servers (AS, Gatan, CEOS).
    """

    def __init__(self):
        super().__init__()
        self.commands = {}

    def connectionMade(self):
        print(f"[Exec] Connection from {self.transport.getPeer()}")

    def register_command(self, name, func):
        """Register a callable command."""
        self.commands[name] = func

    def stringReceived(self, data: bytes):
        msg = data.decode().strip()
        print(f"[Exec] Received: {msg}")
        parts = msg.split()
        cmd, *args = parts

        try:
            if cmd not in self.commands:
                raise ValueError(f"Unknown command: {cmd}")

            handler = self.commands[cmd]
            result = handler(*args)
            if not isinstance(result, (bytes, bytearray)):
                result = str(result).encode()

            self.sendString(result)

        except Exception:
            err = traceback.format_exc()
            print(f"[Exec] Error executing '{msg}':\n{err}")
            self.sendString(err.encode())