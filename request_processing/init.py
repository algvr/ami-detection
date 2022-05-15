import http.server
import socketserver
import ssl
from handler import CustomHTTPRequestHandler


if __name__ == '__main__':
    port = 8144
    handler = CustomHTTPRequestHandler

    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f'Server started at port {port}')
        httpd.socket = ssl.wrap_socket(httpd.socket,
                                       certfile='C:\\Certbot\\live\\algvrithm.com\\fullchain.pem',
                                       keyfile='C:\\Certbot\\live\\algvrithm.com\\privkey.pem', server_side=True)
        httpd.serve_forever()
