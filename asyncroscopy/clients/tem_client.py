# tem_client.py
import socket, struct, numpy as np
from concurrent.futures import ThreadPoolExecutor

# still needs a lot of work
# def send_command() function to generalize
class TEMClient:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=8) # arbitrary for now

    @classmethod
    def connect(cls, host="127.0.0.1", port=9000):
        self.host = host
        self.port = port
        print(f"Connecting to central server {host}:{port}...")
        try:
            with socket.create_connection((host, port), timeout=3) as s:
                print("Connected :)")
            return cls(host, port)
        except (ConnectionRefusedError, socket.timeout):
            print(f"Could not connect to central server at {host}:{port}")
            return None

    def get_image(self, size):
        with socket.create_connection((self.host, self.port)) as s:
            cmd = f"AS_get_image {size}".encode()
            s.sendall(cmd)
            # Read 4-byte length prefix
            hdr = self._recv_exact(s, 4)
            nbytes = struct.unpack("!I", hdr)[0]
            data = self._recv_exact(s, nbytes)
        img = np.frombuffer(data, dtype=np.uint8).reshape(size, size)
        return img

    def get_spectrum(self, size):
        with socket.create_connection((self.host, self.port)) as s:
            cmd = f"Gatan_get_spectrum {size}".encode()
            s.sendall(cmd)
            # Read 4-byte length prefix
            hdr = self._recv_exact(s, 4)
            nbytes = struct.unpack("!I", hdr)[0]
            data = self._recv_exact(s, nbytes)
        spectrum = np.frombuffer(data, dtype=np.float32)
        return spectrum

    def get_image_and_spectrum(self, image_size, spectrum_size):
        """Run both acquisitions concurrently and return results."""
        future_img = self.executor.submit(self.get_image, image_size)
        future_spec = self.executor.submit(self.get_spectrum, spectrum_size)
        # Wait for both to complete
        img = future_img.result()
        spec = future_spec.result()
        return img, spec

    def _recv_exact(self, sock, n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Socket closed early")
            buf += chunk
        return buf