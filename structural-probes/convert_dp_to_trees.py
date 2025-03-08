import networkx as nx


def get_edit_distance(graph_1_edges, graph_2_edges):
    """
    :graph_1_edges: a list of tuples representing the edges in graph 1
    :graph_2_edges: a list of tuples representing the edges in graph 2
    """

    G1 = nx.Graph()
    G1.add_edges_from(graph_1_edges)
    G2 = nx.Graph()
    G2.add_edges_from(graph_2_edges)
    
    print("Graphs successfully constructed, computing edit distance")
    distance = [v for v in nx.optimize_graph_edit_distance(G1, G2)][0]
    print("Done computing edit dist")
    
    return distance

def filter_subtree_edges(subtree_words, edges, words_to_id):
    """
    Given a list of words in a subtree, return the edges only within that subtree
    
    :Parameters:
    - :subtree_words: a list of words in the subtree
    - :edges: a list of tuples representing the edges in the graph
    - :words_to_id: a dictionary mapping words to their indices
    """
    
    subtree_edges = []
    subtree_ids = [words_to_id[word] for word in subtree_words]
    
    for edge in edges:
        if edge[0] in subtree_ids and edge[1] in subtree_ids:
            subtree_edges.append(edge)
    
    return subtree_edges

