import torch
import os
import sys
from pathlib import Path
from graphviz import Digraph

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.gaze_model import GazeNet

def build_manual_diagram():
    """
    Creates a professional looking diagram using Graphviz manually.
    This allows for a cleaner, high-level view suited for papers.
    """
    dot = Digraph(name='GazeNet', comment='GazeNet Architecture', format='png')
    dot.attr(rankdir='TB')
    # Set default node attributes for a professional look
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='white', 
             fontname='Helvetica', fontsize='10', height='0.4')
    dot.attr('edge', fontname='Helvetica', fontsize='10')

    # Inputs
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(style='invis')
        c.node('L_Img', 'Left Eye\n(1x36x60)', shape='oval', fillcolor='#E8F5E9')
        c.node('R_Img', 'Right Eye\n(1x36x60)', shape='oval', fillcolor='#E8F5E9')
        c.node('Head', 'Head Pose\n(2)', shape='oval', fillcolor='#E8F5E9')

    # Helper to create eye stream
    def make_eye_stream(prefix, label):
        # HTML-like label for a compact vertical stack
        # This creates a single node with a table inside, representing the layers
        table = f'''<
        <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
            <TR><TD BORDER="0"><B>{label} Stream</B></TD></TR>
            <TR><TD BGCOLOR="#E3F2FD">Conv1 (32, 3x3)</TD></TR>
            <TR><TD BGCOLOR="#F5F5F5"><FONT POINT-SIZE="9">Pool (2x2)</FONT></TD></TR>
            <TR><TD BGCOLOR="#E3F2FD">Conv2 (64, 3x3)</TD></TR>
            <TR><TD BGCOLOR="#F5F5F5"><FONT POINT-SIZE="9">Pool (2x2)</FONT></TD></TR>
            <TR><TD BGCOLOR="#E3F2FD">Conv3 (128, 3x3)</TD></TR>
            <TR><TD BGCOLOR="#F5F5F5"><FONT POINT-SIZE="9">Pool (2x2)</FONT></TD></TR>
            <TR><TD BGCOLOR="#FFF3E0">FC (128)</TD></TR>
        </TABLE>
        >'''
        
        node_name = f'{prefix}_Net'
        dot.node(node_name, label=table, shape='plain')
        return node_name, node_name

    l_in, l_out = make_eye_stream('L', 'Left Eye')
    r_in, r_out = make_eye_stream('R', 'Right Eye')

    dot.edge('L_Img', l_in)
    dot.edge('R_Img', r_in)

    # Head Stream
    head_table = f'''<
    <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
        <TR><TD BORDER="0"><B>Head Stream</B></TD></TR>
        <TR><TD BGCOLOR="#FFF3E0">FC (32)</TD></TR>
    </TABLE>
    >'''
    dot.node('H_Net', label=head_table, shape='plain')
    
    dot.edge('Head', 'H_Net')

    # Fusion
    dot.node('Concat', 'Concatenate', shape='diamond', fillcolor='#FFF3E0', style='filled')
    
    dot.edge(l_out, 'Concat')
    dot.edge(r_out, 'Concat')
    dot.edge('H_Net', 'Concat')

    # Final Layers
    dot.node('FC1', 'FC\n256')
    dot.node('FC2', 'Output\n(Pitch, Yaw)', shape='oval', fillcolor='#E3F2FD')

    dot.edge('Concat', 'FC1')
    dot.edge('FC1', 'FC2')

    output_path = 'visualization/gaze_net_architecture'
    dot.render(output_path, view=False)
    print(f"Manual diagram saved to {output_path}.png")

if __name__ == '__main__':
    print("Generating manual diagram...")
    build_manual_diagram()

