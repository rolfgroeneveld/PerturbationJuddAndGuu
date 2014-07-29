# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 12:07:34 2014

@author: Rolf
"""

# I wrote this script in Spyder, which by default imports all functionalities
# of NumPy, SciPy, and Matplotlib. On other platforms the following statements
# may be needed:
# from numpy import zeros
# And there might be more I'm not aware of...

# In any case the script definitely needs SymPy for symbolic math
from sympy import *
import matplotlib.pyplot as plt
from numpy import log10 as nplog10

# It is important to distinguish variables representing symbols from
# variables representing values. SymPy does not automatically update a symbol
# variable in an expression when the symbol is assigned a value. To update the
# expression with the assigned value of the variable, the variable must be
# replaced explicitly by the value assigned to it, by either subs or xreplace.
# To keep the distinction between values and symbols clear, I use two 
# different variables for the symbols of the model. What starts with an s 
# is a variable representing a symbol; what starts with a v is a variable
# representing a value. So sx is the variable representing the symbol "x"
# in the symbolic math; vx is the variable representing the numeric value
# of x.

# Initiate symbol variables
sb, sk, sc, skss, sgamma, srho, salpha = \
    symbols('b k c kss gamma rho alpha')

# b:     A dummy variable for equation solving
# k:     The stock of capital
# c:     Consumption
# kss:   The stock of capital in the steady state
# gamma: The coefficient of the utility function
# rho:   The continuous-time discount rate
# alpha: Production elasticity of capital

# n indicates the order of the Taylor polynomial. Be aware that Python starts
# counting at zero, so the polynomial has n+1 terms
n = 5

sa = MatrixSymbol('a',n+1,1)
# a:     Vector of coefficients of the Taylor polynomial

# Value variables of a, rho, alpha, gamma, and kss:
va = zeros(n+1,1)
vrho = 0.05
valpha = 0.25
vgamma = -10
vkss = 1

# Production function
sf = (vrho/valpha)*sk**valpha

# The utility function is not used as such in the code, but it is in the
# Bellman equation implicitly. See the Bellman Equation below, and the
# notes document.

# Construct Taylor polynomial for consumption function
# The consumption function expresses optimal consumption as a function of
# capital stock
sctay = sum([sa[i,0]*(sk-skss)**i/factorial(i) for i in range(n+1)])

# Construct Bellman Equation
# See Judd and Guu and notes document for derivation
sBellman = -vrho*sctay - \
    vgamma*sctay*sctay.diff(sk)+\
    vgamma*sf*sctay.diff(sk)+\
    sctay*sf.diff(sk)

# Find value of a0 from equilibrium condition
# First replace the first vector term by an individual symbol; otherwise the
# condition cannot be solved. This is where we need the dummy sb.
sctay1 = sctay.xreplace({sa[0,0]: sb})
# Then solve this expression for the dummy variable sb:
temp1 = solve(sctay1.subs({sk: skss, skss: vkss}) - sf.subs({sk: vkss}),sb)
# Then substitute the values of rho and alpha to get the value of va0:
va[0] = temp1[0]

# Substitute va[0] in the Bellman Equation
xBellman = sBellman.xreplace({sa[0,0]: va[0]})

# Find first root:
# Find first derivative of Bellman Equation
xBellman = diff(xBellman,sk)
# Solve first derivative Bellman Equation at steady state
temp = solve(xBellman.subs({sk: skss, skss: vkss}).xreplace({sa[1,0]: sb}), sb)
# This yields two different values; assign the positive value to the values vector
va[1] = [x for x in temp if x > 0][0]
# Insert the value in the Bellman Equation
xBellman = xBellman.xreplace({sa[1,0]: va[1]})

# Find other roots:
for i in range(2,n+1):
    # Find first derivative of Bellman Equation
    xBellman = diff(xBellman,sk)
    # Solve first derivative Bellman Equation at steady state
    temp = solve(xBellman.subs({sk: skss, skss: vkss}).xreplace({sa[i,0]: sb}), sb)
    # Higher-order roots are all linear. Assign the value to values vector:
    va[i] = temp[0]
    # Insert value in Bellman Equation
    xBellman = xBellman.xreplace({sa[i,0]: va[i]})

# The polynomial starts from the steady state
sctay = sctay.subs({skss: vkss})

# Insert coefficient values in the polynomial
for i in range(n+1):
    sctay = sctay.xreplace({sa[i,0]: va[i]})

## Make symbolic residual function:
# Copy Bellman equation into residual function
sResid = sBellman
# Replace all symbolic coefficients with their value
for i in range(n+1):
    sResid = sResid.xreplace({sa[i,0]: va[i]})
# Scale residual function by steady-state consumption times rho
sResid = sResid/(vrho*sctay.subs({sk: vkss}))

# Convert symbolic residual function to numeric residual function
fResid = lambdify(sk,sResid.subs({skss: vkss}),"numpy")

# Compute residuals
fResidData = fResid(linspace(.1,2.5,100))

# Plot residuals in logaritmic scale
fig1 = plt.figure()
ax1a = fig1.add_subplot(111)
ax1a.plot(nplog10(fResidData))
