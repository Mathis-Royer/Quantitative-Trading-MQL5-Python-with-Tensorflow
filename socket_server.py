import socket, numpy as np
from sklearn.linear_model import LinearRegression
import os
os.chdir("C:/Users/royer/AppData\/Roaming/MetaQuotes/Terminal/.../MQL5/Include/Hedge_include")
#from dnn-tensorflow import neuronalNetwork

##Class & fonction

class socketserver:
    def __init__(self, address = '', port = 9090):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.address = address
        self.port = port
        self.sock.bind((self.address, self.port))
        self.cummdata = ''

    def recvmsg(self):
        self.sock.listen(1)
        self.conn, self.addr = self.sock.accept()
        print('connected to', self.addr)
        self.cummdata = ''

        while True:
            data = self.conn.recv(10000)
            self.cummdata += data.decode("utf-8")
            if not data:
                break
            self.conn.send(bytes(reponse(self.cummdata), "utf-8"))
            return self.cummdata

    def __del__(self):
        self.sock.close()


def reponse(message):
    return "Message du programme Python"

## Main

serv = socketserver('127.0.0.1', 9090)

while True:
    msg = serv.recvmsg()
    print("message reçu de la part de MQL5 : ", msg)
