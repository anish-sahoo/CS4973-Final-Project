import os
import sys
from graphviz import Digraph

def build_calibration_diagram():
    """
    Creates a diagram for the Calibration model using Graphviz.
    Visualizes the flow from Gaze Vector -> Polynomial Features -> Screen Coordinates.
    """
    dot = Digraph(name='CalibrationModel', comment='Calibration Model Architecture', format='png')
    dot.attr(rankdir='LR') # Left to Right flow
    
    # Set default node attributes for a professional look
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='white', 
             fontname='Helvetica', fontsize='10', height='0.4')
    dot.attr('edge', fontname='Helvetica', fontsize='10')

    # Input Node
    dot.node('Input', 'Gaze Vector\n(Pitch, Yaw)', shape='oval', fillcolor='#E8F5E9')

    # Feature Expansion Step
    expansion_label = '''<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
        <TR><TD BGCOLOR="#E3F2FD"><B>Polynomial Expansion</B></TD></TR>
        <TR><TD>Generate 2nd Degree Terms</TD></TR>
        <TR><TD><I>1, p, y, p·y, p², y²</I></TD></TR>
    </TABLE>
    >'''
    dot.node('Expansion', label=expansion_label, shape='plain')

    # Intermediate Feature Vector
    dot.node('Features', 'Feature Vector\n(6 dimensions)', shape='box', fillcolor='#FFF3E0')

    # Learned Weights (Parameters)
    dot.node('Weights', 'Learned Weights\n(Wx, Wy)', shape='cylinder', fillcolor='#F3E5F5')

    # Training Phase (Where weights come from)
    with dot.subgraph(name='cluster_training') as c:
        c.attr(label='Calibration Phase (One-time)', style='dashed', color='grey')
        c.node('CalibData', 'Calibration Data\n(Gaze Samples + Screen Points)', shape='note', fillcolor='#FFFDE7')
        c.node('Solver', 'Least Squares Solver\n(minimize error)', shape='component', fillcolor='#E0F2F1')
        c.edge('CalibData', 'Solver')
        c.edge('Solver', 'Weights', style='dashed', label=' produces')

    # Regression / Prediction Step
    regression_label = '''<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
        <TR><TD BGCOLOR="#E3F2FD"><B>Linear Regression</B></TD></TR>
        <TR><TD>x = Features · Wx</TD></TR>
        <TR><TD>y = Features · Wy</TD></TR>
    </TABLE>
    >'''
    dot.node('Regression', label=regression_label, shape='plain')

    # Output Node
    dot.node('Output', 'Screen Coordinates\n(Pixels X, Y)', shape='oval', fillcolor='#E8F5E9')

    # Define Edges
    dot.edge('Input', 'Expansion', label=' (p, y)')
    dot.edge('Expansion', 'Features')
    dot.edge('Features', 'Regression')
    dot.edge('Weights', 'Regression', style='dashed', label=' applied')
    dot.edge('Regression', 'Output', label=' (x, y)')

    # Render
    output_path = 'visualization/calibration_model_diagram'
    # Ensure directory exists (it should, but good practice)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        dot.render(output_path, view=False)
        print(f"Calibration diagram saved to {output_path}.png")
    except Exception as e:
        print(f"Error rendering diagram: {e}")
        print("Ensure graphviz is installed on your system (e.g., 'brew install graphviz' on macOS) and the python package is installed ('pip install graphviz').")

if __name__ == '__main__':
    print("Generating calibration diagram...")
    build_calibration_diagram()
