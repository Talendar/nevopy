# MIT License
#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

""" This module implements visualization utilities related to the NEAT
algorithm.
"""

from typing import Any, cast, Dict, List, Optional, Tuple, Union
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import nevopy as ne

# Necessary for forward-reference type-checking.
if TYPE_CHECKING:
    import pygame


def columns_graph_layout(genome: "ne.neat.genomes.NeatGenome",
                         width: float,
                         height: float,
                         node_size: float,
                         horizontal_pad_pc: Tuple[float,
                                                  float] = (.03, .03),
                         vertical_pad_pc: Tuple[float,
                                                float] = (.03, .03),
                         ideal_h_nodes_per_col: int = 4,
                         consider_bias_node: bool = True,
) -> Dict[int, Tuple[float, float]]:
    """ Positions the network's nodes in columns.

    The input nodes are placed in the left-most column and the output nodes
    are placed in the right-most columns. The hidden nodes are placed in
    columns located between those two columns. For big networks, try using a
    smaller node size for better quality.

    Args:
        genome (NeatGenome): The genome to be visualized.
        width (float): Width of the figure / surface.
        height (float): Height of the figure / surface.
        node_size (float): Size of the drawn nodes.
        horizontal_pad_pc (Tuple[float, float]): Tuple containing the size
            of the padding on the left and on the right of the surface.
            Unit: the width of the surface.
        vertical_pad_pc (Tuple[float, float]): Tuple containing the size
            of the padding below and above the surface. Unit: the height of
            the surface.
        ideal_h_nodes_per_col (int): Preferred number of hidden nodes per
            column (the algorithm will try to draw columns with this amount
            of hidden nodes when possible).
        consider_bias_node (bool): Whether the bias node should be
            considered or not when calculating the positions.

    Returns:
        Dictionary mapping the ID of each node to a tuple containing its
        position in the figure.
    """
    pos = {}

    # padding
    origin_x = width * horizontal_pad_pc[0] + node_size / 2
    origin_y = height * vertical_pad_pc[0] + node_size / 2

    width = width - origin_x - horizontal_pad_pc[1] * width - node_size / 2
    height = height - origin_y - vertical_pad_pc[1] * height - node_size / 2

    # procedure for inserting nodes into columns
    def insert_nodes_col(x, nodes):
        """ Inserts the nodes in a column (specified by x). """
        if len(nodes) == 1:
            pos[nodes[0].id] = (x, origin_y + height/2)
            return

        next_y = origin_y
        space_y = height / len(nodes)
        for n in nodes:
            pos[n.id] = (x, next_y + space_y/2)
            next_y += space_y

    # input and bias nodes
    insert_nodes_col(
        x=origin_x,
        nodes=genome.input_nodes + ([] if (genome.bias_node is None
                                           or not consider_bias_node)
                                    else [genome.bias_node]),
    )

    # output nodes
    insert_nodes_col(x=origin_x + width,
                     nodes=genome.output_nodes)

    # hidden nodes
    if genome.hidden_nodes:
        max_num_h_cols = int((width - 4 * node_size) / (2 * node_size))
        h_nodes_per_col = ideal_h_nodes_per_col
        num_cols = max(1, np.ceil(len(genome.hidden_nodes) / h_nodes_per_col))
        while num_cols > max_num_h_cols:
            h_nodes_per_col += 1
            num_cols = np.ceil(len(genome.hidden_nodes) / h_nodes_per_col)

        if num_cols == 1:
            insert_nodes_col(x=width / 2, nodes=genome.hidden_nodes)
        else:
            next_x = origin_x + 2 * node_size
            space_x = (width - 4 * node_size) / num_cols
            h_nodes = np.array(genome.hidden_nodes)
            for node_list in np.array_split(h_nodes, num_cols):
                insert_nodes_col(x=next_x + space_x / 2,
                                 nodes=node_list)
                next_x += space_x

    return pos


def visualize_genome(genome: "ne.neat.genomes.NeatGenome",
                     layout_name: str = "columns",
                     layout_kwargs: Optional[Dict[str, Any]] = None,
                     show: bool = True,
                     block_thread: bool = True,
                     save_to: Optional[str] = None,
                     save_transparent: bool = False,
                     figsize: Tuple[int, int] = (10, 6),
                     node_size: int = 300,
                     pad: int = 1,
                     legends: bool = True,
                     nodes_ids: bool = True,
                     node_id_color: str = "black",
                     edge_curviness: float = 0.1,
                     edges_ids: bool = False,
                     edge_id_color: str = "black",
                     background_color: str = "snow",
                     legend_box_color: str = "honeydew",
                     input_color: str = "deepskyblue",
                     output_color: str = "mediumseagreen",
                     hidden_color: str = "silver",
                     bias_color: str = "khaki",
) -> None:
    """ Plots the neural network (phenotype) encoded by the genome.

    The network is drawn as a graph, with nodes and edges. An edge's color
    is chosen according to the edge's weight. Edges with greater weights are
    drawn with more intense / stronger colors. Edges connecting a node to
    itself aren't be drawn.

    This method uses `NetworkX <https://github.com/networkx/networkx>`_ to
    handle the drawings. It positions the network's nodes according to a
    layout, whose name you can specify in the parameter ``layout_name``. The
    currently available layouts are:

    * All the standard `NetworkX's` layouts available in this
      `link
      <https://networkx.org/documentation/latest/reference/drawing.html#module-networkx.drawing.layout>`_;
    * The `graphviz` layout; it's really good, but to use it you must
      have `Graphviz-Dev` and `pygraphviz` installed on your machine;
    * The `columns` layout (used by default), implemented exclusively for
      `NEvoPy`; it positions the nodes in columns (see
      :meth:`.NeatGenome.columns_graph_layout`, specially the parameter
      ``ideal_h_nodes_per_col``).

    For the colors parameters, it's possible to pass a string with the color
    HEX value or a string with the color's name (names available here:
    https://matplotlib.org/3.1.0/gallery/color/named_colors.html).

    Args:
        genome (NeatGenome): The genome to be visualized.
        layout_name (str): The name of the layout to be used to position the
            network's nodes.
        layout_kwargs (Optional[Dict[str, Any]]): Keyed arguments to be
            passed to the layout. Check each layout documentation for more
            information about the accepted arguments.
        show (bool): Whether to show the generated image or not. If True, a
            window will be created by `matplotlib` to show the image.
        block_thread (bool): Whether to block the execution's thread while
            showing the image. Useful for visualizing multiple networks at
            once. In this case, you should call :meth:`.NeatGenome.visualize`
            with this parameter set to `False` on all genomes except for the
            last one, so all the windows are shown simultaneously.
        save_to (Optional[str]): Path to save the image. If `None`, the
            image won't be automatically saved.
        save_transparent: Whether the saved image should have a transparent
            background or not.
        figsize (Tuple[int, int]): Size of the matplotlib figure.
        node_size (int): Size of the drawn nodes, in `points**2` (the area
            of each node). Default size is 300. See the parameter ``s`` of
            `matplotlib.axes.Axes.scatter
            <https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.axes.Axes.scatter.html>`_
            for more information.
        pad (int): The image's padding (distance between the figure of the
            network and the image's border).
        legends (bool): If `True`, a box with legends describing the nodes
            colors will be drawn.
        nodes_ids (bool): If `True`, the nodes will have their ID drawn
            inside them.
        node_id_color (str): Color of the drawn nodes ids.
        edge_curviness (float): Angle, in radians, of the edges arcs. A
            value of 0 indicates a straight line.
        edges_ids (bool): If `True`, each connection/edge will have its ID
            drawn on it. Keep in mind that some labels might overlap with
            each other, making only one of them visible.
        edge_id_color (str): Color of the drawn connections/edges ids.
        background_color (str): Color of the figure's background.
        legend_box_color (str): Color of the legend box.
        input_color (str): Color of the input nodes.
        output_color (str): Color of the output nodes.
        hidden_color (str): Color of the hidden nodes.
        bias_color (str): Color of the bias node.

    Raises:
        RuntimeError: If both `show` and `save_to` parameters are set to
            `False` (in which case the function wouldn't be doing anything
            but wasting computation).
    """
    # validating args
    if not show and not save_to:
        raise RuntimeError("Both \"show\" and \"save_to\" parameters are "
                           "set to False!")

    # config and start
    plt.rcParams["axes.facecolor"] = background_color
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([n.id for n in genome.nodes()])
    plt.figure(figsize=figsize)

    if layout_kwargs is None:
        layout_kwargs = {}

    # connections
    edges_labels = {}
    for c in genome.connections:
        if c.enabled:
            graph.add_edge(c.from_node.id, c.to_node.id, weight=c.weight)
            edges_labels[(c.from_node.id, c.to_node.id)] = c.id

    # selecting layout
    if layout_name == "graphviz":
        try:
            # pylint: disable=import-outside-toplevel
            # pylint: disable=unused-import
            import pygraphviz
            from networkx.drawing.nx_agraph import graphviz_layout
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Couldn't find the package `pygraphviz`!\nTo draw the "
                "genome's neural network, this package is required. To "
                "install it, however, you first need to install the dev "
                "version of `Graphviz` (https://graphviz.org/download/) on "
                "your system. On Ubuntu, you can do that by executing the "
                "following command:\n"
                "\t~$ sudo apt-get install -y graphviz-dev\n"
                "After installing `Graphviz`, just use pip to install "
                "`pygraphviz` and you're all set:\n"
                "\t~$ pip install pygraphviz") from e

        if "prog" not in layout_kwargs:
            layout_kwargs["prog"] = "dot"
        if "args" not in layout_kwargs:
            layout_kwargs["args"] = "-Grankdir=LR"
        pos = graphviz_layout(graph, **layout_kwargs)

    elif layout_name == "columns":
        plt.xlim(0, figsize[0])
        plt.ylim(0, figsize[1])

        dpi = plt.gcf().dpi
        pos = columns_graph_layout(genome,
                                   *figsize,
                                   node_size=(node_size ** 0.5) / dpi,
                                   **layout_kwargs)
    else:
        nx_layout_func = getattr(nx, layout_name)
        pos = nx_layout_func(graph, **layout_kwargs)

    # plotting
    nx.draw_networkx_nodes(graph, pos=pos,
                           nodelist=[n.id for n in genome.input_nodes],
                           node_size=[node_size] * len(genome.input_nodes),
                           node_color=input_color, label="Input nodes")
    nx.draw_networkx_nodes(graph, pos=pos,
                           nodelist=[n.id for n in genome.output_nodes],
                           node_size=[node_size] * len(genome.output_nodes),
                           node_color=output_color, label="Output nodes")
    nx.draw_networkx_nodes(graph, pos=pos,
                           nodelist=[n.id for n in genome.hidden_nodes],
                           node_size=[node_size] * len(genome.hidden_nodes),
                           node_color=hidden_color, label="Hidden nodes")

    if genome.bias_node is not None:
        nx.draw_networkx_nodes(graph, pos=pos,
                               nodelist=[genome.bias_node.id],
                               node_size=[node_size],
                               node_color=bias_color, label="Bias node")

    if graph.number_of_edges() > 0:
        # calculating edges colors
        edges_weights = list(nx.get_edge_attributes(graph,
                                                    "weight").values())
        if len(edges_weights) == 1:
            edges_colors = [(1, 0.5, 0, 1)]
        else:
            min_w, max_w = np.min(edges_weights), np.max(edges_weights)
            edges_colors = [(1, 0.6 * (1 - (w - min_w) / (max_w - min_w)),
                             0, 0.3 + 0.7 * (w - min_w) / (max_w - min_w))
                            for w in edges_weights]

        # drawing edges
        nx.draw_networkx_edges(
            graph,
            pos=pos,
            edge_color=edges_colors,
            connectionstyle=f"arc3, rad={edge_curviness}"
        )

    if edges_ids:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edges_labels,
                                     font_color=edge_id_color)

    if nodes_ids:
        nx.draw_networkx_labels(graph, pos,
                                labels={k.id: k.id
                                        for k in genome.nodes()},
                                font_size=10,
                                font_color=node_id_color,
                                font_family="sans-serif")
    if legends:
        plt.legend(facecolor=legend_box_color, borderpad=0.8,
                   labelspacing=0.5)

    plt.tight_layout(pad=pad)
    if save_to is not None:
        plt.savefig(save_to, transparent=save_transparent)

    if show:
        plt.show(block=block_thread)


def _nodes_activation_status(genome: "ne.neat.genomes.NeatGenome",
                             hidden_activation_threshold: float,
                             input_visualization_info: Optional[
                                 List["ne.neat.NodeVisualizationInfo"]],
                             output_visualization_info: Optional[
                                 List["ne.neat.NodeVisualizationInfo"]],
                             output_activate_greatest_only: bool,
) -> Dict[int, bool]:
    """ Check each of the network's nodes activation status.

    Returns:
        A :class:`dict` mapping each node's ID to a :class:`bool` (``True``
        if the node is activated and ``False`` otherwise).
    """
    status = {}  # type: Dict[int, bool]
    if output_activate_greatest_only:
        max_out_node = max(genome.output_nodes, key=lambda n: n.activation)
    in_idx = out_idx = 0

    for node in genome.nodes():
        # Bias node:
        if node.type == ne.neat.NodeGene.Type.BIAS:
            status[node.id] = False
        # Output node:
        elif node.type == ne.neat.NodeGene.Type.OUTPUT:
            if output_activate_greatest_only and len(genome.output_nodes) > 1:
                # noinspection PyUnboundLocalVariable
                status[node.id] = node == max_out_node
            else:
                if output_visualization_info is None:
                    status[node.id] = (node.activation
                                       > hidden_activation_threshold)
                else:
                    info = output_visualization_info[out_idx]
                    out_idx += 1
                    status[node.id] = info.is_activated(node.activation)
        # Input node:
        elif node.type == node.type.INPUT:
            if input_visualization_info is None:
                status[node.id] = (node.activation
                                   > hidden_activation_threshold)
            else:
                info = input_visualization_info[in_idx]
                in_idx += 1
                status[node.id] = info.is_activated(node.activation)
        # Hidden node:
        else:
            status[node.id] = node.activation > hidden_activation_threshold

    return status


class NodeVisualizationInfo:
    """ Stores information about an input or output node of a
    :class:`.NeatGenome` to be visualized with the
    :func:`.neat.visualize_activations` function.

    Args:
        label (str): Label to be drawn next to the node.
        activation_threshold (float): Value to be taken as reference when
            checking if the node is activated or not.
        mode (str): Name of the method to be used to check if the node is
            activated or not. Currently available modes: "greater", "less",
            "equal" and "diff".
    """

    _ACTIVATION_MODES = {"greater", "less", "equal", "diff"}

    def __init__(self,
                 label: str = "",
                 activation_threshold: float = 0.5,
                 mode: str = "greater",
                 equality_precision: float = 1e-2) -> None:
        assert mode in NodeVisualizationInfo._ACTIVATION_MODES, (
            f"Invalid mode \"{mode}\" for verifying the activation status of "
            "a node!"
        )

        self.label = label
        self.threshold = activation_threshold
        self.mode = mode
        self.equality_precision = equality_precision

    def is_activated(self, activation: float) -> bool:
        """ Checks whether the node is activated or not. """
        # "Greater than" mode:
        if self.mode == "greater":
            return activation > self.threshold
        # "Less than" mode:
        elif self.mode == "less":
            return activation < self.threshold
        # "Equal" mode:
        elif self.mode == "equal":
            return abs(activation - self.threshold) < self.equality_precision
        # "Different" mode:
        else:
            return abs(activation - self.threshold) > self.equality_precision


# noinspection PyUnboundLocalVariable
def visualize_activations(genome: "ne.neat.genomes.NeatGenome",
                          # Sizes
                          surface_size: Tuple[int, int] = (700, 450),
                          node_radius: float = 14,
                          # Nodes colors
                          node_deactivated_color: Union[
                              str, Tuple[int, int, int]] = (190, 190, 190),
                          node_activated_color: Union[
                              str, Tuple[int, int, int]] = (2, 68, 144),
                          bias_node_color: Union[
                              str, Tuple[int, int, int]] = "yellow",
                          node_border_color: Union[
                              str, Tuple[int, int, int]] = "black",
                          # Edges colors
                          edge_activated_color: Union[
                              str, Tuple[int, int, int]] = (0, 120, 233),
                          edge_deactivated_color: Union[
                              str, Tuple[int, int, int]] = (100, 100, 100),
                          # Edges width
                          activated_edge_width: int = 2,
                          deactivated_edge_width: int = 1,
                          # Padding
                          horizontal_pad_pc: Tuple[float,
                                                   float] = (.015, .015),
                          vertical_pad_pc: Tuple[float,
                                                 float] = (.015, .015),
                          # Nodes labels and activation info
                          hidden_activation_threshold: float = 0.5,
                          input_visualization_info: Optional[Union[
                              List["ne.neat.NodeVisualizationInfo"],
                              List[str],
                          ]] = None,
                          output_visualization_info: Optional[Union[
                              List["ne.neat.NodeVisualizationInfo"],
                              List[str],
                          ]] = None,
                          output_activate_greatest_only: bool = True,
                          # Labels
                          show_input_values: bool = False,
                          show_output_values: bool = False,
                          labels_color: Union[
                                str, Tuple[int, int, int]] = "white",
                          labels_config: Dict[str, Any] = None,
                          # Light
                          show_activation_light: bool = True,
                          activation_light_color: Optional[Union[
                              str, Tuple[int, int, int]]] = (104, 179, 235),
                          activation_light_radius_pc: float = 2,
                          # Others
                          ideal_h_nodes_per_col: int = 4,
                          background_color: Union[
                              str, Tuple[int, int, int]] = "black",
                          node_border_thickness: Optional[float] = 2,
                          draw_bias_node: bool = False,
                          return_rgb_array: bool = False,
) -> Union["pygame.surface.Surface", np.ndarray]:
    """ Draws the network using different colors for activated and deactivated
    nodes and edges.

    Note:
        This method requires :mod:`pygame` installed. You can install it using
        the command:

        .. code::

            $ pip install pygame

    Args:
        genome (NeatGenome): The genome to be visualized.
        surface_size (Tuple[int, int]): Width and height of the pygame
            surface to be drawn.
        node_radius (float): Radius (size) of the drawn nodes.
        node_deactivated_color (Union[str, Tuple[int, int, int]]): Color of
            deactivated nodes.
        node_activated_color (Union[str, Tuple[int, int, int]]): Color of
            activated nodes.
        bias_node_color (Union[str, Tuple[int, int, int]]): Color of the
            bias node.
        node_border_color (Union[str, Tuple[int, int, int]]): Color of the
            nodes' borders.
        edge_activated_color (Union[str, Tuple[int, int, int]]): Color of
            activated edges.
        edge_deactivated_color (Union[str, Tuple[int, int, int]]): Color of
            deactivated edges.
        activated_edge_width (int): The width/thickness of activated edges.
        deactivated_edge_width (int): The width/thickness of deactivated
            edges.
        horizontal_pad_pc (Tuple[float, float]): Tuple containing the size
            of the padding on the left and on the right of the surface.
            Unit: the width of the surface.
        vertical_pad_pc (Tuple[float, float]): Tuple containing the size
            of the padding below and above the surface. Unit: the height of
            the surface.
        hidden_activation_threshold (float): Activation threshold for hidden
            nodes. If the activation value of a hidden node is greater than
            this threshold, the node is considered to be activated.
        input_visualization_info (Optional[Union[
            List[NodeVisualizationInfo], List[str]]]): If it's a list of
            strings, each string will be the label of an input node and default
            settings will be used to determine if an input node is activated or
            not. If it's a list of :class:`.NodeVisualizationInfo` objects, then
            the information provided in the objects will be used instead. If
            ``None``, default settings will be used to determine if an input
            node is activated or not and no labels will be drawn for the input
            nodes.
        output_visualization_info (Optional[Union[
            List[NodeVisualizationInfo], List[str]]]): If it's a list of
            strings, each string will be the label of an output node and default
            settings will be used to determine if an output node is activated or
            not. If it's a list of :class:`.NodeVisualizationInfo` objects, then
            the information provided in the objects will be used instead. If
            ``None``, default settings will be used to determine if an output
            node is activated or not and no labels will be drawn for the output
            nodes.
        show_input_values (bool): If ``True`` and ``input_visualization_info``
            is not ``None``, then the input values will be drawn next to each
            input node.
        show_output_values (bool): If ``True`` and ``output_visualization_info``
            is not ``None``, then the output values will be drawn next to each
            output node.
        output_activate_greatest_only (bool): If ``True``, only one output node
            can be activated at a time (the node with the greatest activation
            value). Otherwise, more than one node can be activated at a time.
        labels_color (Union[str, Tuple[int, int, int]]): Color of the
            labels.
        labels_config (Dict[str, Any]): Keyword arguments to be passed to
            the :meth:`pygame.SysFont` constructor.
        show_activation_light (bool): Whether or not to show a "light ring"
            around activated nodes.
        activation_light_color (Optional[Union[str, Tuple[int, int, int]]]):
            The color of the light ring to be shown around activated nodes.
        activation_light_radius_pc (float): Radius of the light ring to be
            drawn around activated nodes. Unit: the node's radius.
        ideal_h_nodes_per_col (int): Preferred number of hidden nodes per
            column (the algorithm will try to draw columns with this amount
            of hidden nodes whenever possible).
        background_color (Union[str, Tuple[int, int, int]]): The background
            color of the surface.
        node_border_thickness (Optional[float]): Thickness of the nodes'
            borders. If ``None`` or `0`, no border will be drawn.
        draw_bias_node (bool): Whether to draw the network's bias node or
            not.
        return_rgb_array (bool): If ``True``, returns a numpy array with the
            generated image instead of a pygame surface.

    Returns:
        If ``return_rgb_array`` is ``False``, an instance of
        :class:`pygame.Surface` with the drawings is returned. You can
        display it using :mod:`pygame`:

            .. code:: python

                screen_size = 700, 450
                display = pygame.display.set_mode(screen_size)
                # ...
                s = genome.visualize_activations(surface_size=screen_size)
                display.blit(s, [0, 0])
                pygame.display.update()

        If ``return_rgb_array`` is ``True``, a numpy array with the
        generated image is returned instead.

    Raises:
        ModuleNotFoundError: If :mod:`pygame` is not found.
    """
    # Importing pygame:
    try:
        import pygame  # pylint: disable=import-outside-toplevel
        if not pygame.font.get_init():
            pygame.font.init()
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Couldn't find 'pygame'! To use this method, make sure you've "
            "it installed.\nYou can install 'pygame' using pip:\n"
            "\t$ pip install pygame"
        ) from e

    # Creating surface:
    surface = pygame.Surface(surface_size)
    surface.fill(background_color)

    # Font (for rendering the labels):
    if labels_config is None:
        labels_config = dict(name="monospace", size=15,
                             bold=True, italic=False)
    font = pygame.font.SysFont(**labels_config)

    # New variable for the horizontal padding:
    # (required for placing the labels)
    h_pad = list(horizontal_pad_pc)

    # Rendering input labels:
    if input_visualization_info is not None:
        # Processing str list:
        if isinstance(input_visualization_info[0], str):
            input_visualization_info = [
                NodeVisualizationInfo(label=label)  # type: ignore
                for label in input_visualization_info
            ]

        # Casting to list of NodeVisualizationInfo objs:
        input_visualization_info = cast(List[NodeVisualizationInfo],
                                        input_visualization_info)

        assert len(input_visualization_info) == len(genome.input_nodes), (
            "Invalid number of labels or visualization info objects! There "
            "must be one item for each input node."
        )

        # Rendering labels:
        rendered_in_labels = []  # type: List[pygame.surface.Surface]
        for info, node in zip(input_visualization_info,
                              genome.input_nodes):
            # Building the label's text:
            txt = ""
            if len(info.label) > 0:
                txt += f"{info.label}" + (": " if show_input_values else "")

            if show_input_values:
                txt += f"{node.activation:.2f}  -->"

            txt += " " * 4

            # Rendering the label:
            rendered_in_labels.append(
                font.render(txt, True, labels_color)  # type: ignore
            )

        # Adjusting padding:
        max_width = max(rendered_in_labels,
                        key=lambda s: s.get_size()[0]).get_size()[0]
        h_pad[0] = (max_width + h_pad[0] * surface_size[0]) / surface_size[0]

    # Rendering output labels:
    if output_visualization_info is not None:
        # Processing str list
        if isinstance(output_visualization_info[0], str):
            output_visualization_info = [
                NodeVisualizationInfo(label=label)  # type: ignore
                for label in output_visualization_info
            ]

        # Casting to list of NodeVisualizationInfo objs:
        output_visualization_info = cast(List[NodeVisualizationInfo],
                                         output_visualization_info)

        assert len(output_visualization_info) == len(genome.output_nodes), (
            "Invalid number of labels or visualization info objects! There "
            "must be one item for each output node."
        )

        # Rendering labels:
        rendered_out_labels = []  # type: List[pygame.surface.Surface]
        for info, node in zip(output_visualization_info,
                              genome.output_nodes):
            # Building the label's text:
            txt = " " * 3
            if show_output_values:
                txt += f"--> {node.activation:.2f}"

            if len(info.label) > 0:
                txt += " " + info.label

            # Rendering the label:
            rendered_out_labels.append(
                font.render(txt, True, labels_color)  # type: ignore
            )

        # Adjusting padding:
        max_width = max(rendered_out_labels,
                        key=lambda s: s.get_size()[0]).get_size()[0]
        h_pad[1] = (max_width + h_pad[1] * surface_size[0]) / surface_size[0]

    # Checking the activation status of the nodes:
    node_activated = _nodes_activation_status(
        genome=genome,
        hidden_activation_threshold=hidden_activation_threshold,
        input_visualization_info=input_visualization_info,
        output_visualization_info=output_visualization_info,
        output_activate_greatest_only=output_activate_greatest_only,
    )

    # Calculating the nodes' position:
    nodes_pos = columns_graph_layout(
        genome=genome,
        width=surface_size[0],
        height=surface_size[1],
        node_size=2 * node_radius,
        horizontal_pad_pc=h_pad,  # type: ignore
        vertical_pad_pc=vertical_pad_pc,
        ideal_h_nodes_per_col=ideal_h_nodes_per_col,
        consider_bias_node=draw_bias_node,
    )

    # Drawing input labels:
    if input_visualization_info is not None:
        for label, node in zip(rendered_in_labels, genome.input_nodes):
            x, y = nodes_pos[node.id]
            w, h = label.get_size()
            surface.blit(label, dest=(x - w, y - h / 2))

    # Drawing output labels:
    if output_visualization_info is not None:
        for label, node in zip(rendered_out_labels, genome.output_nodes):
            x, y = nodes_pos[node.id]
            h = label.get_size()[1]
            surface.blit(label, dest=(x, y - h / 2))

    # Drawing edges:
    # (must be drawn before the nodes)
    for c in genome.connections:
        if (not c.enabled
            or abs(c.weight * c.from_node.activation) < 1e-3
            or (c.from_node.type == ne.neat.NodeGene.Type.BIAS
                and not draw_bias_node)):
            continue

        if node_activated[c.from_node.id] and c.weight > 0:
            edge_color = edge_activated_color
            edge_width = activated_edge_width
        else:
            edge_color = edge_deactivated_color
            edge_width = deactivated_edge_width

        pygame.draw.line(surface,
                         color=edge_color,
                         start_pos=nodes_pos[c.from_node.id],
                         end_pos=nodes_pos[c.to_node.id],
                         width=edge_width)

    # Drawing nodes:
    for node in genome.nodes():
        if node.type == ne.neat.NodeGene.Type.BIAS and not draw_bias_node:
            continue

        # Border:
        if node_border_thickness is not None and node_border_thickness > 0:
            pygame.draw.circle(surface,
                               color=node_border_color,
                               center=nodes_pos[node.id],
                               radius=node_radius + node_border_thickness)
        # Choosing the node's color:
        color = (bias_node_color if node.type == ne.neat.NodeGene.Type.BIAS
                 else node_activated_color if node_activated[node.id]
                 else node_deactivated_color)
        # Drawing the node:
        pygame.draw.circle(surface,
                           color=color,
                           center=nodes_pos[node.id],
                           radius=node_radius)

    # Drawing activation lights
    if show_activation_light:
        for node in genome.nodes():
            if node_activated[node.id]:
                # Making surface:
                light_radius = node_radius * activation_light_radius_pc
                light_surface = pygame.Surface(size=(2 * light_radius,
                                                     2 * light_radius),
                                               flags=pygame.SRCALPHA)

                # Drawing the circle:
                pygame.draw.circle(light_surface,
                                   color=(node_activated_color
                                          if activation_light_color is None
                                          else activation_light_color),
                                   center=(light_radius, light_radius),
                                   radius=light_radius)

                # Adding alpha gradient to the circle:
                light_array = pygame.surfarray.pixels_alpha(light_surface)

                y = np.linspace(-1, 1, light_array.shape[1])[None, :] * 255
                x = np.linspace(-1, 1, light_array.shape[0])[:, None] * 255

                alpha = np.sqrt(x ** 2 + y ** 2)
                alpha = 255 - np.clip(0, 255, alpha)
                light_array[:] = alpha[:]

                del light_array
                light_surface.unlock()

                # Drawing the generated surface on the main surface:
                surface.blit(source=light_surface,
                             dest=(nodes_pos[node.id][0] - light_radius,
                                   nodes_pos[node.id][1] - light_radius))

    return (surface if not return_rgb_array
            else pygame.surfarray.array3d(surface))
