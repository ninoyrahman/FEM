# -*- coding: utf-8 -*-
# Loads points from a file and create a BSpline
# created by avianse.com
# License: LGPL v 2.1

import FreeCAD
import Part
import Draft
from math import *
import tkinter as tk
from tkinter.filedialog import askopenfilename

root = tk.Tk()
# show askopenfilename dialog without the tkinter window
root.withdraw()

# default is all file types
file_name = askopenfilename()

# Open the file and read the points
# This version expects 1 header/title line, then 2 points per line

points = []
with open(file_name) as f:
    # Remove the header line
    line1 = f.readline()
    # Read the data lines
    for line in f:
        xs, ys = line.split()
        xf=float(xs)
        yf=float(ys)
        # Save this point. I'm treating the points as x and z and setting y to 0
        points.append(FreeCAD.Vector(xf, 0.0, yf))

# Create a curve and then convert to BSpline
curve = Part.makePolygon(points)
Draft.makeBSpline(curve,closed=False,face=False)