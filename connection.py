#!/usr/bin/python

###########################################################################
#
# name          : server.py
#
# purpose       : server to evaluate keywords 
#
# usage         : python server.py
#
# description   : 
#
###########################################################################


import json
# import logger
import socketserver
import sys
import threading
import time

class ThreadedTCPRequestHandler( socketserver.BaseRequestHandler ) :
    def handle( self ) :
        while 1==1:
            try :
                self.data = self.request.recv( 10240 ).strip()
                print ("--> %s wrote:\n%s" % ( self.client_address[ 0 ], str( self.data ) ))
                '''
                response_string = topics.evaluate( str( self.data ) ) 
                print ("--> Evaluation: %s" % response_string)
#                self.request.send( bytes( json.dumps( response_string ), encoding="utf-8" ) )
                self.request.send( str( json.dumps( response_string ) + "\n" ) )
                '''
            except Exception as e :
                break
                logger.log_error( e )


class ThreadedTCPServer( socketserver.ThreadingMixIn, socketserver.TCPServer ) :
    pass


if __name__ == "__main__" :
    HOST = ""
    PORT = 42527
    server = ThreadedTCPServer( ( HOST, PORT ), ThreadedTCPRequestHandler )
    server.serve_forever()
    server_thread = threading.Thread( target=server.serve_forever )
    server_thread.setDaemon( True )
    server_thread.start()
    while True :
        pass

