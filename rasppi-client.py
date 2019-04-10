import socket
import picamera

camera = picamera.PiCamera()

# camera.capture('pict.jpg')

sock = socket.socket()

try:
    sock.connect(('10.1.69.216', 5515))
    conn = sock.makefile('wb')
    # file like object to capture to
    camera.capture(conn, 'jpeg')

    #received = str(sock.recv(1024))
finally: 
    sock.close()
