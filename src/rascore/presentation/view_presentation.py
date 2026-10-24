"""Serve the prebuilt Lectrace presentation locally; Python standard library only."""
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from functools import partial
import argparse,webbrowser
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8782)
    parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent/'site'
    server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(SimpleHTTPRequestHandler,directory=str(root)))
    url=f'http://127.0.0.1:{args.port}/#/?trace=01_route_found&step=1'
    print(f'Lectrace: {url}\nCtrl+C stops this local server. No models are run by the viewer.')
    if not args.no_browser:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()
