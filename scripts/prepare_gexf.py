#!/usr/bin/env python3
import argparse
import datetime

import dateutil.parser
import dateutil.relativedelta
import networkx as nx

import pmer


DATASETS = {
    'dota2': lambda: pmer.datasets.Dota2Dataset.from_csv('dota2.csv'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=DATASETS.keys())
    parser.add_argument('destination', type=argparse.FileType('wb'))
    parser.add_argument('--since',
                        type=dateutil.parser.parse,
                        default=datetime.datetime.now() - dateutil.relativedelta.relativedelta(years=1),
                        help='Start date for events (default is a year ago)')

    args = parser.parse_args()


    prepare_gexf(DATASETS[args.dataset](), args.destination, args.since)


def prepare_gexf(dataset, dst, start_date):
    events = [event for event in dataset if event.date >= start_date]
    graph = construct_game_graph(events)
    nx.write_gexf(graph, dst)


def construct_game_graph(events):
    graph = nx.Graph()

    # Create nodes and edges.
    for event in events:
        for player1_id in event.winners:
            graph.add_node(player1_id, label=str(player1_id), n_games=0)
            for player2_id in event.losers:
                graph.add_node(player2_id, label=str(player2_id), n_games=0)
                graph.add_edge(player1_id, player2_id, weight=0)

    # Add data to nodes and edges.
    for event in events:
        for player1_id in event.winners:
            graph.node[player1_id]['n_games'] += 1
            for player2_id in event.losers:
                graph.node[player2_id]['n_games'] += 1
                graph.edge[player1_id][player2_id]['weight'] += 1

    return graph


main()
