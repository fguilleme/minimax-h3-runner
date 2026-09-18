import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from h3runner.server import Handler, Job, Manager


class ServerVideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.manager = Manager(root / "comfy", root / "runs", root / "server.pid", root / "log")
        output = root / "runs" / "video.mp4"
        output.write_bytes(b"fake-mp4")
        job = Job("abc123", "done", root / "runs" / "job-abc123", output)
        self.manager._jobs[job.id] = job
        Handler.manager = self.manager
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def test_video_endpoint_returns_mp4_bytes(self):
        with urlopen(f"http://127.0.0.1:{self.server.server_port}/video?job=abc123") as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers["Content-Type"], "video/mp4")
            self.assertEqual(response.headers["Content-Disposition"], 'attachment; filename="minimax-h3-abc123.mp4"')
            self.assertEqual(response.read(), b"fake-mp4")

    def test_video_endpoint_rejects_unknown_job(self):
        with self.assertRaises(HTTPError) as context:
            urlopen(f"http://127.0.0.1:{self.server.server_port}/video?job=missing")
        self.assertEqual(context.exception.code, 404)


if __name__ == "__main__":
    unittest.main()