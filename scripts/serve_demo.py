"""Serve the public prebuilt demo using Python's standard library."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8782)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not (ROOT/'docs/lectrace/index.html').exists():
        parser.error('Prebuilt viewer missing. Rebuild the lecture and run scripts/sync_demo.py.')
    handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT/'docs'))
    url = f'http://127.0.0.1:{args.port}/lectrace/#/?trace=01_route_found&step=1'
    with ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
        print(f'Demo: {url}\nPress Ctrl+C to stop.')
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
