"""Serve only the archived V1/V2 comparison; no controller or simulator connection."""

import argparse
import gzip
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit

from scripts.verify_v2_delivery import ROOT,verify


def server(port=8772):
    verify()
    pages={'/':('text/html; charset=utf-8',(ROOT/'workbench/v2/index.html').read_bytes()),
        '/data.json':('application/json; charset=utf-8',gzip.decompress((ROOT/'examples/v2/replay.json.gz').read_bytes())),
        '/health':('application/json',b'{"mode":"archived_readonly","FMU_calls":0}')}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path=urlsplit(self.path).path
            if path not in pages:
                self.send_error(404)
                return
            content_type,data=pages[path]
            self.send_response(200)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store')
            self.end_headers()
            self.wfile.write(data)
        def do_POST(self):
            self.send_error(405,'Archived replay is read-only')
        def log_message(self,*args):
            pass
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8772)
    args=parser.parse_args()
    with server(args.port) as http:
        print(f'Archived comparison only: http://127.0.0.1:{http.server_port}/',flush=True)
        http.serve_forever()
