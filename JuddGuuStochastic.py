# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 12:07:34 2014

@author: Rolf
"""

# I wrote this script in Spyder, which by default imports all functionalities
# of NumPy, SciPy, and Matplotlib. There might be necessary imports I'm not
# aware of

# In any case the script definitely needs SymPy for symbolic math
from sympy import *
from numpy import zeros as npzeros
from scipy.misc import comb
from numpy import log10 as nplog10
from mpl_toolkits.mplot3d import Axes3D

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
sb, sk, sc, skss, sx, sgamma, srho, salpha, ssigma = \
    symbols('b k c kss x gamma rho alpha sigma')

# b:     A dummy variable for equation solving
# k:     The stock of capital
# c:     Consumption
# kss:   The stock of capital in the steady state
# gamma: The coefficient of the utility function
# rho:   The continuous-time discount rate
# alpha: Production elasticity of capital
# sigma: Variance of disturbance z

# n indicates the order of the Taylor polynomial. Be aware that Python starts
# counting at zero, so the polynomial has n+1 terms
n = 4

# Symbolic vectors and matrices
sa = MatrixSymbol('a',n+1,n+1)
# a:     Vector of coefficients of the Taylor polynomial

# Value variables of a, rho, alpha, gamma, and kss:
va = npzeros([n+1,n+1])
powSeriesCoeff = npzeros([n+1,n+1])
vrho = 0.05
valpha = 0.25
vgamma = -10
vkss = 1
vsigma = 0

# Production function
sf = (vrho/valpha)*sk**valpha

# Utility function
su = (sc**(1+sgamma))/(1+sgamma)

# Construct Taylor polynomial for consumption function
tvalue = []

for i in range(n+1):
    for j in range(i+1):
        tvalue.append((1/(factorial(j)*factorial(i-j)))*sa[j,i-j]*(sk-skss)**j*ssigma**(i-j))

sctay = sum(tvalue)

# Auxiliary functions
su1 = simplify(su.diff(sc,1)/su.diff(sc,2)).subs({sc: sctay})
su2 = simplify(su.diff(sc,3)/su.diff(sc,2)).subs({sc: sctay})

# Construct Bellman Equation
# See Judd and Guu and notes document for derivation
sBellman = \
    su1*(sf.diff(sk)-srho)+\
    sctay.diff(sk)*(sf-sctay+2*ssigma*sk)+\
    ssigma*sk**2*(su2*sctay.diff(sk)**2+sctay.diff(sk,2))

# ======== FIRST ROW OF THE COEFFICIENT MATRIX =============================

# Find value of a[0,0] from equilibrium condition
# First replace the first vector term by an individual symbol; otherwise the
# condition cannot be solved. This is where we need the dummy sb.
sctay1 = sctay.xreplace({sa[0,0]: sb})

# Solve this expression for the dummy variable sb:
temp1 = \
    solve(sctay1.subs({sk: skss, skss: vkss, ssigma: 0}) -\
    sf.subs({sk: vkss}),sb)

# Substitute the values of rho and alpha to get the value of va0:
va[0,0] = temp1[0]

# Define temprary Bellman Equation with va[0,0] inserted
xBellman = sBellman.xreplace({sa[0,0]: va[0,0]})

# Find first root:
# Find first derivative of temporary Bellman Equation
xBellman = diff(xBellman,sk)

# Solve first derivative temporary Bellman Equation at steady state
temp = solve(xBellman.subs({sk: skss, skss: vkss, ssigma: 0, sgamma: vgamma,\
    srho: vrho}).xreplace({sa[1,0]: sb}), sb)

# This yields two different values; choose the positive value
# We can safely assume that optimal consumption rises with capital
va[1,0] = [x for x in temp if x > 0][0]

# Insert the value in the temporary Bellman Equation
xBellman = xBellman.xreplace({sa[1,0]: va[1,0]})

# Find other roots:
for i in range(2,n+1):
    # Redefine temporary Bellman Equation as its first derivative with
    # respect to capital
    xBellman = diff(xBellman,sk)
    # Solve new temporary Bellman Equation at steady state
    temp = solve(xBellman.subs({sk: skss, skss: vkss, ssigma: 0,\
        sgamma: vgamma, srho: vrho}).xreplace({sa[i,0]: sb}), sb)
    # Higher-order roots are all linear. Assign the value to values vector:
    va[i,0] = temp[0]
    # Insert value in temporary Bellman Equation
    xBellman = xBellman.xreplace({sa[i,0]: va[i,0]})

# Insert values in actual Bellman Equation
for i in range(n+1):
    sBellman = sBellman.xreplace({sa[i,0]: va[i,0]})
    powSeriesCoeff[i,0] = va[i,0]/factorial(i)

# ======== OTHER ROWS OF COEFFICIENT MATRIX ==============================

for i in range(1,n+1):
    xBellman = diff(sBellman,ssigma,i)
    temp = solve(xBellman.subs({sk: skss, skss: vkss, ssigma: 0,\
        sgamma: vgamma, srho: vrho}).xreplace({sa[0,i]: sb}), sb)
    va[0,i] = temp[0]
    xBellman = xBellman.xreplace({sa[0,i]: va[0,i]})
    for j in range(1,n+1-i):
        xBellman = diff(xBellman,sk)
        temp = solve(xBellman.subs({sk: skss, skss: vkss, ssigma: 0,\
            sgamma: vgamma, srho: vrho}).xreplace({sa[j,i]: sb}), sb)
        va[j,i] = temp[0]
        xBellman = xBellman.xreplace({sa[j,i]: va[j,i]})
    # Insert values in Bellman Equation
    for j in range(n):
        sBellman = sBellman.xreplace({sa[j,i]: va[j,i]})
        powSeriesCoeff[j,i] = va[j,i]/factorial(j)

# The polynomial starts from the steady state
sctay = sctay.subs({skss: vkss})

# Insert coefficient values in the polynomial
for i in range(n+1):
    for j in range(n+1):
        sctay = sctay.xreplace({sa[i,j]: va[i,j]})

fctay = lambdify([sk,ssigma],sctay)

## Make symbolic residual function:
# Copy Bellman equation into residual function
sResid = sBellman

# Replace all symbolic coefficients with their value
for i in range(n+1):
    for j in range(n+1):
        sResid = sResid.xreplace({sa[i,j]: va[i,j]})
        
# Scale residual function by steady-state consumption times rho
sResid = sResid/(vrho*su1.subs({sk: vkss, skss: vkss, ssigma: 0, sgamma: vgamma}).xreplace({sa[0,0]: va[0,0]}))

# Convert symbolic residual function to numeric residual function
fResid = lambdify([sk,ssigma],sResid.subs({skss: vkss, srho: vrho,\
    sgamma: vgamma}),"numpy")

# Compute residuals
kPoints = 100
sPoints = 100
kRange = linspace(.8,1.2,kPoints)
sRange = linspace(0,.001,sPoints)
fResidX = npzeros([kPoints,sPoints])
fResidY = npzeros([kPoints,sPoints])
fResidZ = npzeros([kPoints,sPoints])
for i in range(kPoints):
    fResidX[i,:] = kRange[i]
    fResidY[i,:] = sRange
    fResidZ[i,:] = nplog10(abs(fResid(kRange[i],sRange)))

# Plot residuals in logaritmic scale
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(fResidX, fResidY, fResidZ, rstride=10, cstride=10)

plt.show()

