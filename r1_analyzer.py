"""
A class file for doing analysis of DeepSeek-R1 responses to questions from
OpenAI's MATH500 benchmark dataset.
"""

import os
import re
import math
from collections import defaultdict, Counter
import plotly.colors
import plotly.graph_objects as go
import graphviz
from plotly.subplots import make_subplots

class R1Analyzer:
    """
    A class to analyze responses from DeepSeek-R1.
    """

    def __init__(self):
        """ Initialization method. """
        self.parsed_responses = {}
        self.pattern_types = []
        self.pattern_colors_map = {}
        self.responses_path = None
        self.pattern_regex = re.compile(
            r'\["([^"]+)"\](.*?)(?=\["(?:(?!"end_section")[^"]+)"\]|$)',
            re.DOTALL
        )

    def _update_color_map(self):
        """ Creates or updates the pattern color map for visualizations. """
        colors = plotly.colors.qualitative.Plotly
        num_colors = len(colors)
        self.pattern_colors_map = {
            pattern: colors[i % num_colors]
            for i, pattern in enumerate(self.pattern_types)
        }
        # Add an 'Input' field for the graph's root node
        if 'Input' not in self.pattern_colors_map:
            self.pattern_colors_map['Input'] = 'lightgrey'

    def load_patterns(self, patternfile_path):
        """ (a.k.a., load_stop_words) Loads pattern types from a file. """
        with open(patternfile_path, 'r', encoding='utf-8') as f:
            self.pattern_types = [line.strip() for line in f if line.strip()]
        self._update_color_map()

    def load_responses(self, responses_path, label=None, parser=None):
        """
        (a.k.a, load_text) Loads and parses response text files.
        If a custom parser function is provided, it's used.
        Otherwise we use the default parser.
        """
        # Save the path to the class object
        self.responses_path = responses_path
        # Iterate through the response files
        for file_name in os.listdir(responses_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(responses_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                response_label = label if label else file_name

                # Check if a parser is passed, otherwise use the simple parser as default
                if parser:
                    self.parsed_responses[response_label] = parser(text)
                else:
                    self.parsed_responses[response_label] = self._simple_parser(text)

    def _simple_parser(self, text):
        """ Text cleaning and word counting. Returns a dictionary of word counts. """
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        # Split into words
        words = text.split()
        # Return dictionary of word counts
        return dict(Counter(words))

    def parse_pattern(self, text):
        """ Parses patterns from response text (pattern_name, pattern_content). """
        # Initalize data list
        parsed_data = []
        # Find matches for our patterns
        matches = self.pattern_regex.finditer(text)
        # Iterate through all our matched patterns
        for match in matches:
            pattern_name = match.group(1).strip()
            # Make sure we ignore the end_section labels
            if pattern_name.lower() != 'end_section' and pattern_name in self.pattern_types:
                pattern_content = match.group(2).strip()
                parsed_data.append((pattern_name, pattern_content))
        return parsed_data

    def _calculate_proportions(self, parsed_data):
        """ Calculates the proportion of characters for each pattern. """
        pattern_chars = defaultdict(int)
        total_chars = 0
        # Iterate through the data and count the characters
        for pattern, content in parsed_data:
            char_count = len(content)
            pattern_chars[pattern] += char_count
            total_chars += char_count

        proportions = {}
        # Calculate the proportions, handle possible division by zero
        for pattern in self.pattern_types:
            proportions[pattern] = pattern_chars.get(pattern, 0) / total_chars if total_chars > 0 else 0.0

        return proportions

    def get_response_keys(self):
        """ Returns a sorted list of response file names. """
        return sorted(self.parsed_responses.keys())

    def make_sankey(self, response_keys=None):
        """ Make a sankey diagram of responses and patterns. """
        if response_keys is None:
            response_keys = self.get_response_keys()

        # Setup the labels and indices
        labels = response_keys + self.pattern_types
        source_indices = {key: i for i, key in enumerate(response_keys)}
        pattern_indices = {pattern: i + len(response_keys) for i, pattern in enumerate(self.pattern_types)}

        # Initialize lists
        source, target, value, opaque_link_colors = [], [], [], []

        # Generate file colors
        num_files = len(response_keys)
        base_file_colors = plotly.colors.qualitative.Plotly
        num_base_colors = len(base_file_colors)
        file_colors = [base_file_colors[i % num_base_colors] for i in range(num_files)]

        node_colors = file_colors + [self.pattern_colors_map.get(p, 'grey') for p in self.pattern_types]

        # Iterate through the responses to set up the sankey nodes and edges for each response
        for i, resp_key in enumerate(response_keys):
            parsed_data = self.parsed_responses[resp_key]
            proportions = self._calculate_proportions(parsed_data)
            total_chars = sum(len(content) for _, content in parsed_data)

            source_idx = source_indices[resp_key]
            source_color = file_colors[i]

            for pattern, proportion in proportions.items():
                if proportion > 0:
                    target_idx = pattern_indices[pattern]
                    source.append(source_idx)
                    target.append(target_idx)
                    value.append(proportion * total_chars)
                    opaque_link_colors.append(source_color)

        # Setup color transparency for a clearer diagram
        transparent_link_colors = []
        for c in opaque_link_colors:
            hex_val = c.lstrip('#')
            rgb = tuple(int(hex_val[i:i+2], 16) for i in (0, 2, 4))
            transparent_link_colors.append(f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.3)')

        # Setup the figure
        fig = go.Figure(data=[go.Sankey(
            node={
                "pad": 15, "thickness": 20, "line": {"color": "black", "width": 0.5},
                "label": labels, "color": node_colors
            },
            link={
                "source": source, "target": target, "value": value,
                "color": transparent_link_colors, "hovercolor": opaque_link_colors
            }
        )])
        fig.update_layout(title_text="Pattern Proportion Per R1 Chain-of-Thought", font_size=12)
        fig.show()

    def make_pattern_histogram(self, response_keys=None):
        """ Generates histogram for each pattern. """
        if response_keys is None:
            response_keys = self.get_response_keys()

        # Organize the subplots into columns.
        num_files = len(response_keys)
        num_cols = 2
        num_rows = math.ceil(num_files / num_cols)

        fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=response_keys)

        # Setup the histograms for each response
        for i, resp_key in enumerate(response_keys):
            parsed_data = self.parsed_responses[resp_key]
            proportions_dict = self._calculate_proportions(parsed_data)

            patterns_in_order = self.pattern_types
            proportions_in_order = [proportions_dict.get(p, 0.0) for p in patterns_in_order]
            bar_colors = [self.pattern_colors_map.get(p, 'grey') for p in patterns_in_order]

            # Variables for rows and columns
            current_row = (i // num_cols) + 1
            current_col = (i % num_cols) + 1

            # Create the histogram
            fig.add_trace(go.Bar(
                x=patterns_in_order,
                y=proportions_in_order,
                marker_color=bar_colors,
                name=resp_key
            ), row=current_row, col=current_col)

            fig.update_yaxes(
                title_text="Proportion",
                range=[0, 0.6],
                row=current_row,
                col=current_col
            )

        # Title and styling
        fig.update_layout(
            title_text="Pattern Proportion per R1 Chain-of-Thought",
            height=400*num_rows,
            showlegend=False
        )
        fig.show()

    def make_average_histogram(self, response_keys=None):
        """ Generates a single histogram of average proportions. """
        if response_keys is None:
            response_keys = self.get_response_keys()

        # Initialize utility variables
        pattern_proportion_sums = defaultdict(float)
        included_files_count = 0

        for resp_key in response_keys:
            parsed_data = self.parsed_responses[resp_key]
            total_chars = sum(len(content) for _, content in parsed_data)
            if total_chars > 0:
                proportions = self._calculate_proportions(parsed_data)
                for pattern, proportion in proportions.items():
                    pattern_proportion_sums[pattern] += proportion
                included_files_count += 1

        # Create the average proportion dictionary for the histogram bars
        average_proportions = {
            pattern: pattern_proportion_sums.get(pattern, 0.0) / included_files_count
            for pattern in self.pattern_types
        }

        # Store lists for the histogram configuration
        patterns_list = self.pattern_types
        avg_values = [average_proportions.get(p, 0.0) for p in patterns_list]
        bar_colors = [self.pattern_colors_map.get(p, 'grey') for p in patterns_list]

        # Setup the histogram
        fig = go.Figure(data=[go.Bar(x=patterns_list, y=avg_values, marker_color=bar_colors)])
        fig.update_layout(
            title_text=f"Average Pattern Proportion Across {included_files_count} Chains-of-Thought",
            xaxis_title="Behavioral Pattern",
            yaxis_title="Average Proportion",
            title_font={"size": 36, "family": "Arial Black"},
            font={"size": 24, "family": "Arial Black"},
            yaxis_range=[0, max(avg_values or [0]) * 1.1]
        )
        fig.update_xaxes(
            title_font={"size": 30, "family": "Arial Black"},
            tickfont={"size": 24, "family": "Arial Black"}
        )
        fig.update_yaxes(
            title_font={"size": 30, "family": "Arial Black"},
            tickfont={"size": 24, "family": "Arial Black"}
        )
        fig.show()

    def make_trajectory_graph(self, question_prefix: str):
        """ Generates a tree where nodes are patterns and edges are weighted by trajectory occurrence. """
        relevant_sequences = []

        # Find the relevant files that match the prefix and get the corresponding chain-of-thought pattern sequence.
        for fname, parsed_data in self.parsed_responses.items():
            if fname.startswith(question_prefix):
                sequence = [pattern for pattern, _ in parsed_data]
                if sequence:
                    relevant_sequences.append(sequence)

        # Initialize the tree structure
        tree_root = {
            'pattern': 'Input',
            'count': len(relevant_sequences),
            'id': 'input_root',
            'children': {}
        }
        node_counter = 0

        # Build the tree by traversing the sequence and adding nodes
        for seq in relevant_sequences:
            current_node = tree_root
            for pattern_name in seq:
                if pattern_name in self.pattern_types:
                    if pattern_name not in current_node['children']:
                        # Create a new node if this pattern hasn't been seen at this position.
                        node_counter += 1
                        # Clean the name so it's safe to use for the graphviz ID.
                        safe_pattern_name = re.sub(r'\W+', '_', pattern_name).strip('_')
                        new_node = {
                            'pattern': pattern_name,
                            'count': 1,
                            'id': f'{safe_pattern_name}_{node_counter}',
                            'children': {}
                        }
                        current_node['children'][pattern_name] = new_node
                        current_node = new_node
                    else:
                        # If the pattern already exists at this position, increment it's count for the edge weight.
                        child_node = current_node['children'][pattern_name]
                        child_node['count'] += 1
                        current_node = child_node

        # Initialize graphviz digraph
        dot = graphviz.Digraph(comment=f'Pattern Trajectory Tree for {question_prefix}')
        dot.attr(rankdir='TB')

        # Graph creation (bfs)
        queue = [(None, tree_root)]
        while queue:
            parent_tree_node, current_tree_node = queue.pop(0)
            node_id = current_tree_node['id']
            node_label = current_tree_node['pattern']
            node_count = current_tree_node['count']
            # Boxes for pattern nodes but ellipse for the input node.
            node_shape = 'box' if node_label != 'Input' else 'ellipse'
            # Set the node color based on the pattern type.
            node_color = self.pattern_colors_map.get(node_label, 'lightgrey')
            # Add the node to the graph.
            display_label = f"{node_label}"
            dot.node(node_id, label=display_label, shape=node_shape, style='filled', fillcolor=node_color)
            # Add the edge from parent to current node.
            if parent_tree_node:
                parent_id = parent_tree_node['id']
                edge_label = str(node_count)
                penwidth = '1'
                dot.edge(parent_id, node_id, label=edge_label, penwidth=penwidth)
            # Add children to the queue
            for child_node in current_tree_node['children'].values():
                queue.append((current_tree_node, child_node))

        # Render and save the graph
        output_filename = f'{question_prefix}_trajectory'
        dot.render(output_filename, view=True, cleanup=True, format='pdf')

if __name__ == "__main__":

    # Setup variables
    PATTERN_FILE = "patternfile.txt"
    DATA_DIR = "data"
    QUESTION_FOR_GRAPH = 'q1'

    # Initialize the class & data methods
    analyzer = R1Analyzer()
    analyzer.load_patterns(PATTERN_FILE)
    analyzer.load_responses(DATA_DIR, parser=analyzer.parse_pattern)
    all_response_keys = analyzer.get_response_keys()

    # Visualization calls
    analyzer.make_sankey(all_response_keys)
    analyzer.make_pattern_histogram(all_response_keys)
    analyzer.make_average_histogram(all_response_keys)
    analyzer.make_trajectory_graph(QUESTION_FOR_GRAPH)
