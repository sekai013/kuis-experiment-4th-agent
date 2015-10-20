import sys
import socket
import click

import svm

class AuctionClient(object):

    BUFSIZE = 4096

    def __init__(self, options):
        self.host = options['host']
        self.port = options['port']

    def connect(self):
        # connect to auction server
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))

        # ask username after all participants connected
        click.echo(self.socket.recv(self.BUFSIZE).rstrip())
        self.username = raw_input()
        self.socket.send('%s\n' % self.username)
        click.echo('Welcome to auction, %s.\n' % self.username)
        click.echo('Waiting for other participants...\n')

        # receive auction info after all participants register username

        ## participants
        participants = self.socket.recv(self.BUFSIZE).rstrip()

        ## my userid
        userid = self.socket.recv(self.BUFSIZE).rstrip()
        self.userid = int(userid.split(' ')[-1])

        ## the number of participants and goods
        data = self.socket.recv(self.BUFSIZE).rstrip()
        participant_size, item_size = map(\
                lambda s : int(s.split(' ')[-1]),\
                data.split('\n'))

        click.echo('%d clients are taking part in this auction:'\
                % participant_size)
        click.echo(participants)
        click.echo('%d items will be exibited.' % item_size)
        click.echo('Your ID is a%d.' % self.userid)
        click.echo(self.socket.recv(self.BUFSIZE).rstrip())

        self.start_bid()

    def start_bid(seld):
        'Bid'


# Settings for CLI app
CONTEXT_SETTINGS = { 'help_option_names': ['-h', '--help'] }

@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('host', nargs=1, required=True)
@click.argument('port', nargs=1, required=True)
def cli(host, port):
    
    options = {
            'host': host,
            'port': int(port)
            }

    client = AuctionClient(options)
    client.connect()


if __name__ == '__main__':
    
    if len(sys.argv) < 3:
        click.echo('  Usage: python auction_client.py HOST PORT')
        exit()

    options = {}
    options['host'] = sys.argv[1]
    options['port'] = int(sys.argv[2])

    client = AuctionClient(options)

    client.connect()
