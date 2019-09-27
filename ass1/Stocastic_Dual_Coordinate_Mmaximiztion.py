import numpy as np
import random as rnd
import time as tm
import matplotlib.pyplot as plt
# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length
def getCyclicCoord( currentCoord ,n):
    if currentCoord >= n-1 or currentCoord < 0:
        return 0
    else:
        return currentCoord + 1

def getRandCoord( currentCoord,n ):
    return rnd.randint( 0, n-1 )

def getRandpermCoord( currentCoord ,n):
    global randperm, randpermInner
    if randpermInner >= n-1 or randpermInner < 0 or currentCoord < 0:
        randpermInner = 0
        randperm = np.random.permutation( y.size )
        return randperm[randpermInner]
    else:
        randpermInner = randpermInner + 1
        return randperm[randpermInner]
# Get the CSVM objective value in order to plot convergence curves
def getCSVMObjVal( X,y,C,theta ):
    w = theta[0:-1]
    b = theta[-1]
    hingeLoss = np.maximum( 1 - np.multiply( (X.dot( w ) + b), y ), 0 )
    return 0.5 * w.dot( w ) + C * hingeLoss.dot(hingeLoss)

def getCSVMObjValDual( C,alpha, w, b ):
    # Recall that b is supposed to be treated as the last coordinate of w
    return np.sum(alpha) - 0.5 * np.square( np.linalg.norm( w ) ) - 0.5 * b * b - (0.25/C)*np.square(np.linalg.norm(alpha))

    
        
################################
# Non Editable Region Starting #
################################
def solver( X, y, C, timeout, spacing ):
	(n, d) = X.shape
	t = 0
	totTime = 0
	
	# w is the normal vector and b is the bias
	# These are the variables that will get returned once timeout happens
	w = np.zeros( (d,) )
	b = 0
	tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################
	ticPlotting=tm.perf_counter()
	primalObjValSeries = []
	dualObjValSeries = []
	alpha = C*np.ones((y.size))
	alphay = np.multiply( alpha, y )
	# Initialize the model vector using the equations relating primal and dual variables
	w = X.T.dot( alphay )
    # Recall that we are imagining here that the data points have one extra dimension of ones
    # This extra dimension plays the role of the bias in this case
	b = alpha.dot( y )
    # Calculate squared norms taking care that we are appending an extra dimension of ones
	normSq = np.square( np.linalg.norm( X, axis = 1 ) ) + 1
	i = -1
	totTimePlotting=0
	timeSeries=[]
    
	# You may reinitialize w, b to your liking here
	# You may also define new variables here e.g. eta, B etc

################################
# Non Editable Region Starting #
################################
	while True:
		t = t + 1
		if t % spacing == 0:
			print(timeSeries[-1],primalObjValSeries[-1],dualObjValSeries[-1])
		
			toc = tm.perf_counter()
			totTime = totTime + (toc - tic)
			if totTime > timeout:
				fig,axs=plt.subplots(2)
				axs[0].plot( timeSeries,primalObjValSeries, color = 'b', linestyle = '-', label = "SDCM Primal" )
				axs[1].plot( timeSeries, dualObjValSeries, color = 'b', linestyle = ':', label = "SDCM Dual" )
				plt.show()	
				return (w, b, totTime)
			else:
				tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################
		i = getRandCoord( i ,n)
		x = X[i,:]
        # Find the unconstrained new optimal value of alpha_i
		newAlphai =  (1+alpha[i]*normSq[i] - y[i] * (x.dot(w) + b))/(normSq[i]+1/2*C)
#         newAlphai=alpha[i] + (1-y[i] * (x.dot(w) + b))/normSq[i]
        # Make sure that the constraints are satisfied
		if newAlphai > C:
		    newAlphai = C
		if newAlphai < 0:
			newAlphai = 0
        
        # Update the model vector and bias values
        # Takes only O(d) time to do so :)
		w = w + (newAlphai - alpha[i]) * y[i] * x
		b = b + (newAlphai - alpha[i]) * y[i]
        
		alpha[i] = newAlphai

		tocPlotting = tm.perf_counter()
		totTimePlotting = (tocPlotting - ticPlotting)
        
		primalObjValSeries.append(getCSVMObjVal( X,y,C,np.append( w, b ) ))
		dualObjValSeries.append(getCSVMObjValDual( C,alpha, w, b ))
		timeSeries.append(totTimePlotting)
		# Write all code to perform your method updates here within the infinite while loop
		# The infinite loop will terminate once timeout is reached
		# Do not try to bypass the timer check e.g. by using continue
		# It is very easy for us to detect such bypasses - severe penalties await
		
		# Please note that once timeout is reached, the code will simply return w, b
		# Thus, if you wish to return the average model (as we did for GD), you need to
		# make sure that w, b store the averages at all times
		# One way to do so is to define two new "running" variables w_run and b_run
		# Make all GD updates to w_run and b_run e.g. w_run = w_run - step * delw
		# Then use a running average formula to update w and b
		# w = (w * (t-1) + w_run)/t
		# b = (b * (t-1) + b_run)/t
		# This way, w and b will always store the average and can be returned at any time
		# w, b play the role of the "cumulative" variable in the lecture notebook
		# w_run, b_run play the role of the "theta" variable in the lecture notebook
		
	return (w, b, totTime) # This return statement will never be reached