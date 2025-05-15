import networkx as nx
import matplotlib.pyplot as plt

class LanguageGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_language(self, lang):
        self.graph.add_node(lang)

    def add_translation_edge(self, src, tgt, quality):
        self.graph.add_edge(src, tgt, weight=quality)

    def get_neighbors(self, lang):
        return list(self.graph.successors(lang))

    def get_edge_quality(self, src, tgt):
        return self.graph[src][tgt]['weight']

    def get_all_languages(self):
        return list(self.graph.nodes)

    def display_graph(self):
        print("Languages and translation paths:")
        for src, tgt, attr in self.graph.edges(data=True):
            print(f"{src} -> {tgt}, Quality: {attr['weight']:.2f}")

import numpy as np
import random

class QLearningAgent:
    def __init__(self, graph, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.graph = graph
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def initialize_q_table(self):
        for src in self.graph.get_all_languages():
            self.q_table[src] = {}
            for tgt in self.graph.get_neighbors(src):
                self.q_table[src][tgt] = 0.0

    def choose_action(self, current_lang):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.graph.get_neighbors(current_lang))
        else:
            return max(self.q_table[current_lang], key=self.q_table[current_lang].get)

    def update_q_value(self, current, next_lang, reward):
        max_future_q = max(self.q_table[next_lang].values(), default=0)
        old_q = self.q_table[current][next_lang]
        new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
        self.q_table[current][next_lang] = new_q

    def get_optimal_path(self, start_lang, end_lang, max_steps=10):
        path = [start_lang]
        current = start_lang
        steps = 0
        while current != end_lang and steps < max_steps:
            if not self.q_table.get(current):
                break
            current = self.choose_action(current)
            path.append(current)
            steps += 1
        return path if path[-1] == end_lang else None


def simulate_translation_quality(path, graph):
    if not path or len(path) < 2:
        return 0.0
    total_quality = 1.0
    for i in range(len(path) - 1):
        src, tgt = path[i], path[i + 1]
        edge_quality = graph.get_edge_quality(src, tgt)
        total_quality *= edge_quality
    return total_quality


def main():
    graph = LanguageGraph()
    languages = ['en', 'fr', 'de', 'es', 'it']

    for lang in languages:
        graph.add_language(lang)

    # Add translation quality edges (simulated or real data based)
    graph.add_translation_edge('en', 'fr', 0.9)
    graph.add_translation_edge('fr', 'de', 0.85)
    graph.add_translation_edge('de', 'es', 0.8)
    graph.add_translation_edge('es', 'it', 0.9)
    graph.add_translation_edge('en', 'de', 0.6)
    graph.add_translation_edge('en', 'es', 0.7)
    graph.add_translation_edge('en', 'it', 0.5)

    graph.display_graph()

    agent = QLearningAgent(graph)
    agent.initialize_q_table()

    # Train Q-learning
    for _ in range(1000):
        current = 'en'
        end = 'it'
        steps = 0
        while current != end and steps < 10:
            next_lang = agent.choose_action(current)
            reward = graph.get_edge_quality(current, next_lang)
            agent.update_q_value(current, next_lang, reward)
            current = next_lang
            steps += 1

    optimal_path = agent.get_optimal_path('en', 'it')
    quality = simulate_translation_quality(optimal_path, graph)
    print(f"\nOptimal Path: {' -> '.join(optimal_path)}")
    print(f"Estimated Quality: {quality:.4f}")

def draw_translation_graph(graph, optimal_path=None):
    G = graph.graph
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes and all edges
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1500)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='-|>', arrowsize=15, edge_color='gray')

    # Draw edge labels (translation quality)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')

    # Highlight the optimal path if given
    if optimal_path and len(optimal_path) > 1:
        path_edges = list(zip(optimal_path, optimal_path[1:]))
        nx.draw_networkx_edges(
            G, pos,
            edgelist=path_edges,
            edge_color='red',
            width=2.5,
            arrowstyle='-|>',
            arrowsize=20
        )

    plt.title("Translation Language Graph and Optimal Path")
    plt.axis('off')
    plt.show()
def draw_translation_graph(graph, optimal_path=None, filename="translation_graph.png"):
    G = graph.graph
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes and all edges
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1500)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='-|>', arrowsize=15, edge_color='gray')

    # Draw edge labels (translation quality)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')

    # Highlight optimal path
    if optimal_path and len(optimal_path) > 1:
        path_edges = list(zip(optimal_path, optimal_path[1:]))
        nx.draw_networkx_edges(
            G, pos,
            edgelist=path_edges,
            edge_color='red',
            width=3.0,
            arrowstyle='-|>',
            arrowsize=20
        )

    plt.title("Translation Language Graph with Optimal Path")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, format="png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()

