import sys
import socket
import click
import itertools
import os
from datetime import date

import svm

class AuctionClient(object):

    BUFSIZE = 4096

    LOG_DIR = 'auction_log'

    def __init__(self, options):
        self.host = options['host']
        self.port = options['port']
        self.demo = options['demo']

        self.bid_round = 0
        self.prices = []
        self.histories = []

    def connect(self):
        # connect to auction server
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))

        # ask username after all participants connected
        click.echo('Waiting for other participants...\n')
        click.echo(self.socket.recv(self.BUFSIZE).rstrip())
        click.echo('name> ', nl=False)
        self.username = 'hoge' if self.demo else raw_input()
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
        self.participant_size, self.item_size = map(\
                lambda s : int(s.split(' ')[-1]),\
                data.split('\n'))

        self.prices.append(['0'] * self.item_size)
        self.winners = ([None] * self.item_size)

        click.echo('%d clients are taking part in this auction:'\
                % self.participant_size)
        click.echo(participants)
        click.echo('%d items will be exibited.' % self.item_size)
        click.echo('Your ID is a%d.' % self.userid)
        click.echo('%s\n' % self.socket.recv(self.BUFSIZE).rstrip())

        self.start_bid()

    def start_bid(self):
        click.echo('bid> ', nl=False)

        if self.demo > 1:
            bid = '1' * self.item_size
            self.demo = self.demo - 1
            click.echo(bid)
        elif self.demo == 1:
            bid = '0' * self.item_size
            self.demo = self.demo - 1
            click.echo(bid)
        else:
            bid = raw_input()

        self.socket.send('%s\n' % bid)
        click.echo('Waiting for other participants...')

        result = self.socket.recv(self.BUFSIZE).rstrip()
        history = map(lambda x : list(x.split(':')[-1]), \
                result.split(' ')[-self.participant_size:])
        self.histories.append(history)

        click.echo(result)

        msg = self.socket.recv(self.BUFSIZE).rstrip()

        if msg == 'end':
            click.echo('''
*****************
* Auction Ended *
*****************
''')
            self.end_auction()
        else:
            price = map(lambda x : x.split(':')[-1], \
                    msg.split(' '))
            self.prices.append(price)

            oldprice = self.prices[self.bid_round]

            for i in range(self.item_size):
                if oldprice[i] == price[i] and self.winners[i] == None:
                    self.winners[i] = [j for h, j in\
                            zip(history, range(len(history)))\
                            if h[i] == '1']

            self.bid_round += 1

            click.echo('%s\n' % msg)
            self.start_bid()

    def end_auction(self):
        winner = self.socket.recv(self.BUFSIZE).rstrip()
        price = self.socket.recv(self.BUFSIZE).rstrip()
        self.socket.recv(self.BUFSIZE).rstrip()
        self.socket.recv(self.BUFSIZE).rstrip()
        result = self.socket.recv(self.BUFSIZE).rstrip()
        self.socket.close()

        self.winner = map(lambda s : s[-1], winner.split(' ')[2:])

        line_length = self.item_size * 2 \
                + (self.item_size + 1) * self.participant_size
        half_line = '-' * (line_length / 2)

        click.echo(winner)
        click.echo('%s\n' % price)
        click.echo('%s result %s' % (half_line, half_line))
        click.echo(result)
        click.echo('-' * (line_length + 8))

        self.format_result(result)

    def format_result(self, result, max_level=3):

        if not os.path.exists(self.LOG_DIR):
            os.mkdir(self.LOG_DIR)

        save_dir = '%s/%s' % (self.LOG_DIR, date.today())

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        item_range = range(self.item_size)

        for winner, item_index in zip(self.winners, item_range):
            if winner == None:
                continue
            for w in winner:
                for h in self.histories:
                    h[w][item_index] = '1'

      # for winner, i in zip(self.winner, range(len(self.winner))):
      #     for h in self.histories:
      #         h[int(winner)-1][i] = '1'

        for i in range(1, max_level+1):
            for j in range(self.participant_size):
                if j+1 == self.userid:
                    continue

                user_dir = '%s/participant_%d' % (save_dir, j+1)

                if not os.path.exists(user_dir):
                    os.mkdir(user_dir)

                histories_j = map(lambda h : h[j], self.histories)

                for combi in itertools.combinations(item_range, i):
                    combi_prices = map(lambda row :\
                            [x for (x, k) in zip(row, range(len(row)))\
                            if k in combi], self.prices)

                    combi_histories = map(lambda row :\
                            [x for (x, k) in zip(row, range(len(row)))\
                            if k in combi], histories_j)

                    combi_class = map(lambda row :\
                            ['1'] if len(set(row)) == 1 and '1' in\
                            set(row) else ['-1'],\
                            combi_histories)

                    save_data = '\n'.join(\
                            map(lambda row: ' '.join(row[0] + row[1]),\
                            zip(combi_prices, combi_class)))

                    save_path = '%s/item-%s.dat' %\
                            (user_dir, '-'.join(map(\
                            lambda i : str(i+1), combi)))

                    with open(save_path, 'a+') as save_file:
                        save_file.write('%s\n' % save_data)


# Settings for CLI app
CONTEXT_SETTINGS = { 'help_option_names': ['-h', '--help'] }

@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('host', nargs=1, required=True)
@click.argument('port', nargs=1, required=True)
@click.option('-d', '--demo', default=0, help='bid automatically for n times.')
def cli(host, port, demo):
    
    options = {
            'host': host,
            'port': int(port),
            'demo': demo
            }

    client = AuctionClient(options)
    client.connect()


if __name__ == '__main__':
    
    if len(sys.argv) < 3:
        click.echo(' Usage: python auction_client.py HOST PORT [DEMO]')
        click.echo('   DEMO: bid automatically for n times.')
        exit()

    options = {}
    options['host'] = sys.argv[1]
    options['port'] = int(sys.argv[2])
    options['demo'] = 0 if len(sys.argv) < 4 else int(sys.argv[3])

    client = AuctionClient(options)
    client.connect()
