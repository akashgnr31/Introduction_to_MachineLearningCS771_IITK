import numpy as np
import random as rnd
import time as tm
from matplotlib import pyplot as plt
import random


# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length
def getCyclicCoord( currentCoord, n ):
    if currentCoord >= n-1 or currentCoord < 0:
        return 0
    else:
        return currentCoord + 1

def getRandCoord( currentCoord,n ):
    return random.randint( 0, n-1 )

def getRandpermCoord( currentCoord, randperm, randpermInner, n, y):
    # global randperm, randpermInner
    if randpermInner >= n-1 or randpermInner < 0 or currentCoord < 0:
        randpermInner = 0
        randperm = np.random.permutation( y.size )
        return randperm[randpermInner]
    else:
        randpermInner = randpermInner + 1
        return randperm[randpermInner]

# Get the CSVM objective value in order to plot convergence curves
def getCSVMObjVal(X, y , C , theta ):
    w = theta[0:-1]
    b = theta[-1]
    hingeLoss = np.maximum( 1 - np.multiply( (np.dot(X,w) + b), y ), 0 )
    return 0.5 * np.dot(w,w) + C * np.dot(hingeLoss,hingeLoss)

def getCSVMObjValDual( alpha, w, b, C ):
    # Recall that b is supposed to be treated as the last coordinate of w
    return np.sum(alpha) - 0.5 * np.square( np.linalg.norm( w ) ) - 0.5 * b * b - (0.25/C)*np.square(np.linalg.norm(alpha))

def getStepLength( t,eta ):
    return eta/(t+1)

def mySVM( X ):
    return np.dot(X,w) + b

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
	tic_plotting=tm.perf_counter()
	theta=np.append(w,b)
	cumulative=np.zeros((d+1,))
	primalObjValSeries=[]
	dualObjValSeries=[]
	validationPrimalObjValue=[]
	validationDualObjValue=[]
	timeSeries=[]
	totTimeplotting=0
	# You may reinitialize w, b to your liking here
	# You may also define new variables here e.g. eta, B etc

	# primalObjValSeries = np.zeros( (horizon,) )
	# dualObjValSeries = np.zeros( (horizon,) )
	# timeSeries = np.zeros( (horizon,) )
	# totTime = 0
	alpha = C * np.ones( (y.size,) )
	alphay = np.multiply( alpha, y )
	w = np.dot(np.transpose(X),alphay)
	b = np.dot(alpha,y)
	normSq = np.square( np.linalg.norm( X, axis = 1 ) ) + 1
	i = -1
	# theta1=np.append(w,b)
	# print(getCSVMObjVal(X,y,C,theta1))
	# print(getCSVMObjValDual(alpha, w, b, 1))
################################
# Non Editable Region Starting #
################################
	while True:
		t = t + 1
		if t % spacing == 0:
			print(t)
			toc = tm.perf_counter()
			totTime = totTime + (toc - tic)
			# print(t,totTime,getCSVMObjVal,dualObjValSeries[-1])
		
			if totTime > timeout:
				# fig,axs=plt.subplots(2)
				# axs[0].plot(timeSeries,primalObjValSeries)
				# axs[0].set_title("Primal Loss Traiing Data")
				# axs[1].plot(timeSeries,dualObjValSeries)
				# axs[1].set_title("Dual Loss Training Data")
				# axs[2].plot(timeSeries,validationPrimalObjValue)
				# axs[2].set_title("Primal Loss Test Data")
				# axs[3].plot(timeSeries,validationDualObjValue)
				# axs[3].set_title("Dual Loss Test Data")
				# plt.show()
				
				return (w, b, totTime)
			else:
				tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################
		# print(w)
		randperm = np.random.permutation( y.size )
		randpermInner = -1		
		C = 1
		# eta = 300000
		eta = 24560

		i = getRandCoord(i,n)
		# i = getCyclicCoord(i,n)
		# i = getRandpermCoord( i,randperm, randpermInner, n, y )
		x=X[i,:]
		
		delAlphai = 1 - (alpha[i]/C) - (y[i]*(x.dot(w) + b))
		# print((x.dot(w) , b))
		newAlphai =  alpha[i] + getStepLength(t ,eta) * delAlphai
		# print(newAlphai,alpha[i],i)

		# print("Alpha = "+ str(newAlphai)+" "+str(i))

		if newAlphai > C:
			newAlphai = C
		if newAlphai < 0:
			newAlphai = 0
			
		w = w + np.multiply((newAlphai - alpha[i]) * y[i],x)
		b = b + (newAlphai - alpha[i]) * y[i]
		alpha[i] = newAlphai

		theta=np.append(w,b)      
		# # cumulative = cumulative + theta
		# toc = tm.perf_counter()
		# totTimeplotting = toc - tic_plotting
		# if t%1000==0:
		# 	print(t,totTimeplotting,getCSVMObjVal(X,y,C,theta))
		# primalObjValSeries.append(getCSVMObjVal(X,y,C,theta))
		# dualObjValSeries.append(getCSVMObjValDual(alpha, w, b, C))
		# timeSeries=np.append(timeSeries,totTimeplotting)
		
		# print("validation")
		# validationPrimalObjValue.append(getCSVMObjVal(X_test,y_test,C,theta))
		# validationDualObjValue.append(getCSVMObjValDual(alpha, w, b, C))
		# print(t,totTimeplotting,validationPrimalObjValue[-1])
		# print(t,totTimeplotting,validationDualObjValue[-1])
		
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
