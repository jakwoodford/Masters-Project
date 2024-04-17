import ROOT
from root_numpy import root2array, tree2array
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import stats
import pickle
from scipy.optimize import leastsq as least_squares
from scipy.optimize import curve_fit

#note in order to get root_numpy we need to be in python 2 so printing is a little odd

def numpyDivide(a,b):
    return np.divide(a,b,out=np.zeros_like(a), where=b!=0)

#remember to run 'source lcgenv.sh' to set up root, python etc

#Defns for fit func and chi-sq

def fitFunc(p, x):
    '''
    Fit function
    '''
    f= p[0]*x+p[1]

    return f

def fitFuncDiff(p, x):
    '''
    Differential of fit function
    '''
    df= p[0]
    return df

def calcChiSq(p, x, y, xerr, yerr):
    '''
    Error function for fit
    '''
    e = (y - fitFunc(p, x))/(np.sqrt(yerr**2 + fitFuncDiff(p, x)**2*xerr**2))
    return e


def fitStdError(jacMatrix):

    # Compute covariance
    jMat2 = np.dot(jacMatrix.T, jacMatrix)
    detJmat2 = np.linalg.det(jMat2)
    
    # Prepare output
    output = np.zeros(jMat2.shape[0])
    if detJmat2 < 1E-32:
        print("Value of determinat detJmat2",detJmat2)
        print("Matrix singular, error calculation failed.")
        return output
    else:
        covar = np.linalg.inv(jMat2)
        for i in range(len(output)):
            output[i] = np.sqrt(covar[i, i])
            
        return output

#input file path
path = '/bundle/data/MuEDM/PhysicsProject2024/ForJak/'

#number corresponds to different generated settings
inputFileNum = 120
chain = ROOT.TChain("t1")

#setup directories for writing plots
outputDirTop = "Plots/" + str(inputFileNum) + "/"  #All plots
histDir = "Histograms/" #All histograms

#nfiles to read
nfiles = 500
for fNum in range(1,nfiles):
    if fNum%50 == 0:
        print("still running...", fNum)

    if fNum == 300:
        continue    
    fNumStr = format(fNum, "03")

    filename = 'musr_'+str(inputFileNum)+'_simple_17EDM_noMS_'+ fNumStr +'.root' #EDM = 0.0
    chain.Add(path+filename)

n_events = chain.GetEntries()

print("Total number of events in input files:", n_events)

#to just do a quick scan uncomment here                                                                                                                     
#n_events = 10000                                                                                                                                           
print("Total number of events to run over:", n_events)

#these are the names of the variables read in from the input file
branchNames=['det_n', 'runID', 'det_x', 'det_y', 'det_z', 'muDecayPosX', 'muDecayPosY', 'muDecayPosZ', 'muDecayTime', 'muDecayPolX', 'muDecayPolY', 'muDecayPolZ', 'posIniMomX', 'posIniMomY', 'posIniMomZ', 'eventID', 'weight', 'timeToNextEvent', 'BFieldAtDecay', 'muIniTime', 'muIniPosX', 'muIniPosY', 'muIniPosZ', 'muIniMomX', 'muIniMomY', 'muIniMomZ', 'muIniPolX', 'muIniPolY', 'muIniPolZ', 'muDecayDetID', 'det_ID', 'det_edep', 'det_edep_el', 'det_edep_pos', 'det_edep_gam', 'det_edep_mup', 'det_nsteps', 'det_length', 'det_time_start', 'det_time_end', 'det_kine']

array = tree2array(chain, branches=branchNames, start=0, stop=n_events)

##################################################################################################################################################
##################################################################################################################################################


nBins = 50

#A - all, B - 0 to half p, C - half p to E_max
p_max = 125.0

#place momentum cuts here
p_lim = [(0, p_max), (0, p_max/5.0), (p_max/5, 2*p_max/5), (2*p_max/5, 3*p_max/5), (3*p_max/5, 4*p_max/5), (4*p_max/5, p_max)]
momDir = ["momDirA/", "momDirB/", "momDirC/", "momDirD/", "momDirE/", "momDirF/"] 

amp = []
ampError = []
freq = []
freqError = []

ampFixed = []
ampErrorFixed =[]

AVals = []
AValsErr = []
BVals = []
BValsErr = []

freqMod = []
freqModErr = []

fitDataMax = []

array1 = []

#i is the index, mDir the actual name
for i, mDir in enumerate(momDir): 
    
    #add momentum dir to output directory
    outputDir = outputDirTop + mDir
    print "Output Directory: {}".format(outputDir)

    #apply momentum cuts, defined in p_lim
    desk1 = (np.where(np.sqrt(array['posIniMomX']**2 + array['posIniMomY']**2 + array['posIniMomZ']**2) <= p_lim[i][1]))
    desk2 = (np.where(np.sqrt(array['posIniMomX']**2 + array['posIniMomY']**2 + array['posIniMomZ']**2) > p_lim[i][0]))
    desk = np.intersect1d(desk1, desk2)
    desk3 = np.where(array['muDecayTime'] < 30)
    desk = np.intersect1d(desk, desk3)

    #this is all the variables but with the cut above applied
    array1 = array[desk]  

    print "selecting momentum between {} MeV, with {} entries".format(p_lim[i], len(desk))

    #must be a better way of doing this - maybe a dictionary?
    tot_det_n           = array1[branchNames[0]]
    tot_runID           = array1[branchNames[1]]
    tot_det_x           = array1[branchNames[2]]
    tot_det_y           = array1[branchNames[3]]
    tot_det_z           = array1[branchNames[4]]
    tot_decay_pos_x     = array1[branchNames[5]]
    tot_decay_pos_y     = array1[branchNames[6]]
    tot_decay_pos_z     = array1[branchNames[7]]
    tot_muDecayTime     = array1[branchNames[8]]
    tot_muDecayPolX     = array1[branchNames[9]]
    tot_muDecayPolY     = array1[branchNames[10]]
    tot_muDecayPolZ     = array1[branchNames[11]]
    tot_posIniMomX      = array1[branchNames[12]]
    tot_posIniMomY      = array1[branchNames[13]]
    tot_posIniMomZ      = array1[branchNames[14]]
    tot_eventID         = array1[branchNames[15]]
    tot_weight          = array1[branchNames[16]]
    tot_timeToNextEvent = array1[branchNames[17]]
    #tot_BFieldAtDecay   = array1[branchNames[18]]
    tot_muIniTime       = array1[branchNames[19]]
    tot_muIniPosX       = array1[branchNames[20]]
    tot_muIniPosY       = array1[branchNames[21]]
    tot_muIniPosZ       = array1[branchNames[22]]
    tot_muIniMomX       = array1[branchNames[23]]
    tot_muIniMomY       = array1[branchNames[24]]
    tot_muIniMomZ       = array1[branchNames[25]]
    tot_muIniPolX       = array1[branchNames[26]]
    tot_muIniPolY       = array1[branchNames[27]]
    tot_muIniPolZ       = array1[branchNames[28]]
    tot_muDecay_DetID   = array1[branchNames[29]]
    tot_det_ID          = array1[branchNames[30]]
    tot_det_edep        = array1[branchNames[31]]
    tot_edep_el         = array1[branchNames[32]]
    tot_edep_pos        = array1[branchNames[33]]
    tot_det_edep_gam    = array1[branchNames[34]]
    tot_det_edep_mup    = array1[branchNames[35]]
    tot_det_nsteps      = array1[branchNames[36]]
    tot_det_length      = array1[branchNames[37]]
    tot_det_time_start  = array1[branchNames[38]]
    tot_det_time_end    = array1[branchNames[39]]
    tot_det_kine        = array1[branchNames[40]]

    #Plot of total momentum of positron
    plt.clf()
    tot_posIniMomX_array = np.array(tot_posIniMomX)
    tot_posIniMomY_array = np.array(tot_posIniMomY)
    tot_posIniMomZ_array = np.array(tot_posIniMomZ)
    tot_posIniMom_array = np.sqrt(tot_posIniMomX_array**2 + tot_posIniMomY_array**2 + tot_posIniMomZ_array**2)

    tot_decay_pos_x_array = np.array(tot_decay_pos_x)
    tot_decay_pos_y_array = np.array(tot_decay_pos_y)
    tot_decay_pos_z_array = np.array(tot_decay_pos_z)
     
    counts, binEdges = np.histogram(tot_posIniMom_array, bins = nBins, range = (0, p_max))
    binCentres = (binEdges[:-1] + binEdges[1:]) / 2
    err = np.sqrt(counts)
    countsArr = np.repeat(max(counts), nBins)
    #for k in range(nBins):
        #countsNorm = counts[k]/countsArr[k]
        #print(countsNorm)
    print(countsArr)
    countsNorm = counts/(countsArr)
    print(counts)
    #print(max(counts))
    print(countsNorm)
    errNorm = err/max(counts)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.errorbar(binCentres, counts, yerr=err, fmt='.')
    plt.xlabel("Positron Momentum [MeV/c]")
    plt.ylabel("Number of Events")
    #plt.axvline(x = 0, linestyle = '--', color = 'g')
    #plt.axvline(x = 25, linestyle = '--', color = 'g', label = 'Momentum Bins')
    #plt.axvline(x = 50, linestyle = '--', color = 'g')
    #plt.axvline(x = 75, linestyle = '--', color = 'g')
    #plt.axvline(x = 100, linestyle = '--', color = 'g')
    #plt.axvline(x = 125, linestyle = '--', color = 'g')
    alpha = 0.2
    ax.axvspan(0, 25, alpha = alpha, label = '0 - 25 MeV/c', color = 'g')
    ax.axvspan(25, 50, alpha = alpha, label = '25 - 50 MeV/c', color = 'r')
    ax.axvspan(50, 75, alpha = alpha, label = '50 - 75 MeV/c')
    ax.axvspan(75, 100, alpha = alpha, label = '75 - 100 MeV/c', color = 'm')
    ax.axvspan(100, 125, alpha = alpha, label = '100 - 125 MeV/c', color = 'y')
    plt.legend()
    plt.savefig(outputDir + 'positron_mom.png')    


    #decay time
    #plt.clf()
    countsTime, binEdgesTime = np.histogram(tot_muDecayTime, bins = nBins) 
    binCentresTime = (binEdgesTime[:-1] + binEdgesTime[1:]) / 2
    errCounts = np.sqrt(countsTime, where=countsTime!=0)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.errorbar(binCentresTime, countsTime, yerr=errCounts, fmt='.')
    #ax.set_yscale("log")
    plt.xlabel('Time ($\mu$s)')
    plt.ylabel('Number of decays')
    plt.savefig(outputDir + 'decay_time.png')

    #decay time fit

    xdata = binCentresTime
    ydata = np.log(countsTime, where=countsTime!=0)
    xerror = 0
    #yerror = errCounts*1/countsTime
    yerror = np.divide(errCounts, countsTime, out=np.ones_like(errCounts), where=countsTime!=0)

    #print(counts)
    #print(err)
    #print(yerror)

    initParams = [1,1]
    pInit = initParams
    lBounds = -100*np.ones(len(initParams))
    uBounds = 100*np.ones(len(initParams))
    nPoints = len(xdata)
    nPars = len(initParams)

    # Run fit
    #output = least_squares(calcChiSq, pInit, args = (xdata, ydata, xerror, yerror))
    output = least_squares(calcChiSq, pInit, args = (xdata, ydata, xerror, yerror), full_output=True)
    x = output[0]
    cov_x = output[1]                

    print(x)
    print(cov_x)
    print(len(output))

    # Get least_squares output, stored in array output.x[]
    m = x[0]
    c = x[1]


    # Get errors from our fits using fitStdError(), defined above
    pErrors = cov_x #fitStdError(output.jac)
    d_m = np.sqrt(pErrors[0][0])
    d_c = np.sqrt(pErrors[1][1])

    # Calculate fitted y-values using our fit parameters and the original fit function
    xPlot = np.linspace(np.min(xdata), np.max(xdata), 300)
    fitData = fitFunc(x, xPlot)   

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #ax.set_yscale("log")
    plt.errorbar(xdata, ydata, yerr=yerror, fmt='.', label = 'Simulated Data')
    plt.plot(xPlot, fitData, color = 'r', label = "Fit")
    plt.xlabel('Time ($\mu$s)')
    plt.ylabel('ln(Number of decays)')
    plt.legend(numpoints = 1)
    plt.savefig(outputDir + 'decay_time_fit.png')      

    # Output fit parameters
    print("Fitted parameters: m = {0:.4f}, c = {1:.2f}".format(m, c))
    #print("Parameter errors: m = {1:.2f}, c = {1:.2f}".format(d_m, d_c)


    # Calculate chis**2 per point, summed chi**2 and chi**2/NDF
    chiarr = calcChiSq(x, xdata, ydata, xerr = xerror, yerr = yerror)**2
    chisq = np.sum(chiarr)
    NDF = nPoints - nPars
    chisqndf = chisq/NDF
    print("ChiSq = {:5.2e}, ChiSq/NDF = {:5.2f}.".format(chisq, chisqndf))  

    muonLifetime = -1/m 
    muonLifetimeErr = (m**-2)*d_m 

    print("Muon lifetime = ({:5.2f} +/- {:5.2f})mu s".format(muonLifetime, muonLifetimeErr)) 

    muonLifetimeLit = 2.19703
    muonLifetimeLitErr = 0.00004
    gammaFactor = muonLifetime/muonLifetimeLit  
    gammaFactorErr = np.sqrt((muonLifetimeLitErr/muonLifetime)**2+(muonLifetimeLit/muonLifetime**2)**2*muonLifetimeErr**2)

    print "Relativistic gamma factor =", gammaFactor
    print "Relativistic gamma factor error =", gammaFactorErr

    muonMass = 105.7 #MeV
    momCalc = muonMass*np.sqrt(gammaFactor**2 - 1)

    print('\n')
    print(momCalc)
    print('\n\n\n')

    #x-z plane
    fig = plt.figure()
    plt.scatter(tot_decay_pos_x_array, tot_decay_pos_z_array)
    ax.set_yscale("log")
    plt.xlabel('Decay Position in x-plane [m]')
    plt.ylabel('Decay Position in z-plane [m]')
    plt.savefig(outputDir + 'decay_pos_xz.png')

    #x-y plane
    fig = plt.figure()
    plt.scatter(tot_decay_pos_x_array, tot_decay_pos_y_array)
    ax.set_yscale("log")
    plt.xlabel('Decay Position in x-plane [m]')
    plt.ylabel('Decay Position in y-plane [m]')
    plt.savefig(outputDir + 'decay_pos_xy.png')

    #Decay angle Plots

    decayAngle = np.arcsin(tot_posIniMomZ_array/tot_posIniMom_array)
    decayAngleDeg = np.arcsin(tot_posIniMomZ_array/tot_posIniMom_array)*180/np.pi

    print(max(decayAngle))
    
    #decayAngleAvg = np.sum(decayAngle)/len(decayAngle)
    #print(decayAngleAvg)

    countsAngle, binEdgesAngle = np.histogram(decayAngle, bins = nBins) 
    binCentresAngle = (binEdgesAngle[:-1] + binEdgesAngle[1:]) / 2
    errCountsAngle = np.sqrt(countsAngle)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.errorbar(binCentresAngle, countsAngle, yerr=errCountsAngle, fmt='.')
    plt.title('Momentum Between {0:.0f} MeV and {1:.0f} MeV'.format(p_lim[i][0], p_lim[i][1]))
    plt.xlabel('Decay Angle [rad]')
    plt.ylabel('Number of Events')
    plt.savefig(outputDir + 'decay_angle_rad.png')

    mean = np.average(binCentresAngle, weights = countsAngle)
    var = np.average((binCentresAngle - mean)**2, weights = countsAngle)
    widths = np.sqrt(var)

    print('Width of decay angles:', widths)
    print('\n')

    countsAngleDeg, binEdgesAngleDeg = np.histogram(decayAngleDeg, bins = nBins) 
    binCentresAngleDeg = (binEdgesAngleDeg[:-1] + binEdgesAngleDeg[1:]) / 2
    errCountsAngleDeg = np.sqrt(countsAngleDeg)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.errorbar(binCentresAngleDeg, countsAngleDeg, yerr=errCountsAngleDeg, fmt='.')
    plt.xlabel('Decay Angle [Degrees]')
    plt.ylabel('Number of events')
    plt.savefig(outputDir + 'decay_angle_deg.png')


    #Decay angle Gaussian fit
    
    def fitGauss(p, x):
        '''
        Fit function
        '''
        f= p[0]*np.exp(-(1/2)*((x-p[1])/p[2])**2)+p[3]
        return f

    def fitFuncDiffGauss(p, x):
        '''
        Differential of fit function
        '''
        df= -p[0]*np.exp(-(1/2)*((x-p[1])/p[2])**2)*((x-p[1])/(p[2]**2))
        return df

    def calcChiSqGauss(p, x, y, xerr, yerr):
        '''
        Error function for fit
        '''
        e = (y - fitGauss(p, x))/(np.sqrt(yerr**2 + fitFuncDiffGauss(p, x)**2*xerr**2))
        return e    

    #Initial Guesses
    '''
    a = 1200
    mu = 0.2
    w = 1
    c = 10

    xData = binCentresAngle
    yData = countsAngle
    xError = 0
    yError = errCountsAngle

    initialParams = np.array([a, mu, w, c])
    parInit = initialParams
    #lBounds = -100*np.ones(len(initParams))
    #uBounds = 100*np.ones(len(initParams))
    nPoints1 = len(xData)
    nPars1 = len(initialParams)

    #print(yError)

    #Run fit

    outputGauss = least_squares(calcChiSqGauss, parInit, args = (xData, yData, xError, yError), full_output=True)
    xGauss = outputGauss[0]
    cov_xGauss = outputGauss[1]
    
    print(xGauss)
    print('\n')
    print(cov_xGauss)

    # Get least_squares output, stored in array output.x[]
    a = xGauss[0]
    b = xGauss[1]
    c = xGauss[2]
    d = xGauss[3]

    # Get errors from our fits using fitStdError(), defined above
    
    pErrors = cov_xGauss #fitStdError(output.jac)
    d_A = np.sqrt(pErrors[0][0], where=pErrors[0][0]!=0)
    d_b = np.sqrt(pErrors[1][1])
    d_c = np.sqrt(pErrors[2][2])
    d_d = np.sqrt(pErrors[3][3])
    
    xPlot1 = np.linspace(np.min(xData), np.max(xData), 300)
    fitData1 = fitGauss(xGauss, xPlot1)

    # Make the plot of the data and the fit
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.errorbar(xData, yData, yerr=yError, fmt='.', label = 'Data')
    plt.plot(xPlot1, fitData1, color = 'r', label = "Fit")
    plt.xlabel('Decay Angle [rad]')
    plt.ylabel('Number of events')
    plt.legend()
    plt.savefig(outputDir + 'decay_angle_fit.png') 


    # Output fit parameters
    print("Fitted parameters: A= {0:.2f}, mu= {1:.2f}, sigma= {2:.2f}, c= {3:.2f}".format(a, b, c, d))
    print("Parameter errors: A= {0:.2f}, mu= {1:.2f}, sigma= {2:.2f}, c= {3:.2f}".format(d_A, d_b, d_c, d_d))


    # Calculate chis**2 per point, summed chi**2 and chi**2/NDF
    chiarr1 = calcChiSqGauss(xGauss, xData, yData, xError, yError)**2
    chisq1 = np.sum(chiarr1)
    NDF1 = nPoints1 - nPars1
    chisqndf1 = chisq1/NDF1
    print("ChiSq = {:5.2e}, ChiSq/NDF = {:5.2f}.".format(chisq1, chisqndf1))
    '''

    sumAnglesTime, binEdgesSumAnglesTime = np.histogram(tot_muDecayTime, bins = nBins, weights = decayAngle)
    b = binEdgesSumAnglesTime
    binCentresSumAnglesTime = (binEdgesSumAnglesTime[:-1] + binEdgesSumAnglesTime[1:]) / 2
    binCentresSumAnglesTime2 = []

    for j,v in enumerate(b):
        if v == b[-1]:
            break
        gt = muonLifetimeLit * gammaFactor
        a =  gt * ( np.exp(-b[j+1]/gt) - np.exp(-b[j]/gt) ) / (-b[j+1] + b[j])
        binCentresSumAnglesTime2.append(a + b[j])

    #print("times:")                                                                                                                                                                                                      
    #print(b)
    #print(binCentresSumAnglesTime)
    #print(binCentresSumAnglesTime2)
    binCentresSumAnglesTime = np.array(binCentresSumAnglesTime2)

    errsumAnglesTime = np.sqrt(np.histogram(tot_muDecayTime, bins = nBins, weights = decayAngle**2)[0])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.errorbar(binCentresSumAnglesTime, sumAnglesTime, yerr=errsumAnglesTime, fmt='.')
    plt.title('Momentum Between {0:.0f} MeV and {1:.0f} MeV'.format(p_lim[i][0], p_lim[i][1]))                                                                                                                                 
    plt.xlabel('Time ($\mu$s)')
    plt.ylabel('Sum of Decay Angles [rad]')
    plt.savefig(outputDir + 'sumAngles_time.png')

    #print(np.histogram(tot_muDecayTime, bins = 20, weights = decayAngle)[0])
    #print(np.histogram(tot_muDecayTime, bins = 20, weights = decayAngle**2)[0])
    print("Error on sum of angles:", errsumAnglesTime)
    print('\n')

    avgAngleTime = np.divide(sumAnglesTime, countsTime, where=countsTime!=0)
    #avgAngleTimeErr = np.sqrt(((1/countsTime)**2)*(errsumAnglesTime**2) + ((-sumAnglesTime/(countsTime)**2)**2)*(errCounts**2))
    #widthArr = np.array([1.18, 1.2, 1.58, 0.935, 1.175, 0.65])
    avgAngleTimeErr =  np.divide(widths, errCounts, where=errCounts!=0)
    #avgAngleTimeErr2 =  widthArr[i]/errCounts
    #havgAngleTime, edges = np.histogram(avgAngleTime, bins = 20)                                                                                           
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.errorbar(binCentresSumAnglesTime, avgAngleTime, yerr=avgAngleTimeErr, fmt='.') 
    plt.title('Momentum Between {0:.0f} MeV and {1:.0f} MeV'.format(p_lim[i][0], p_lim[i][1]))                                                                                                                                 
    plt.xlabel('Time [$\mu$s]')
    plt.ylabel('Average Decay Angle [rad]')
    plt.savefig(outputDir + 'avgAngles_time.png')


    print("Number of counts:", countsTime)
    print("Error on sum of angles:", errsumAnglesTime)
    print("Sum angle values:", sumAnglesTime)
    print("Error on number of counts:",errCounts)
    #print("")
    print("Error on average:", avgAngleTimeErr)
    print("implied spread in angles from error:", avgAngleTimeErr*errCounts)

    def fitSine(p, x):
        '''
        sine wave
        '''
        f = p[0] + p[1]*np.sin((p[2]*x) + p[3]) 
        return f

    def fitSineDiff(p, x):
        '''
        Differential of sine fit
        '''
        df = p[1]*p[2]*np.cos(p[2]*x + p[3])
        return df       

    def calcChiSqSine(p, x, y, xerr, yerr):
        '''
        Error function for fit
        '''
        e = (y - fitSine(p, x))/(np.sqrt(yerr**2 + fitSineDiff(p, x)**2*xerr**2))
        return e 

    #Fixed frequency and zero phase      

    def fitSine2(p, x):
        '''
        sine wave
        '''
        w = 0.37
        f = p[0] + p[1]*np.sin((w*x) + 0) 
        return f

    def fitSineDiff2(p, x):
        '''
        Differential of sine fit
        '''
        w = 0.37
        df = p[1]*w*np.cos(w*x + 0)
        return df       

    def calcChiSqSine2(p, x, y, xerr, yerr):
        '''
        Error function for fit
        '''
        e = (y - fitSine2(p, x))/(np.sqrt(yerr**2 + fitSineDiff2(p, x)**2*xerr**2))
        return e 

    def fitSineMod(p, x):
        '''
        Modified sine wave 
        '''
        f1 = p[1]*np.sin((p[2]*x) + p[3])
        f2 = 1 + p[4]* np.cos((p[2]*x + p[3]))
        return (f1/f2) + p[0]

    def fitSineModDiff(p, x):
        '''
        Differential of modified sine fit
        '''
        df = p[1] * p[2] * ( p[4] + np.cos(p[2]*x + p[3]) / (p[4] * np.cos(p[2]*x + p[3])+1)**2)
        return df

    def calcChiSqSineMod(p, x, y, xerr, yerr):
        '''
        Error function for fit
        '''
        e = (y - fitSineMod(p, x))/(np.sqrt(yerr**2 + fitSineModDiff(p, x)**2*xerr**2))
        return e 

    #Initial Guesses
    
    C = 0.0
    A = 0.1
    omega = 0.40
    p = 0.0

    xDataSine = binCentresSumAnglesTime
    yDataSine = avgAngleTime
    xErrorSine = 0
    yErrorSine = avgAngleTimeErr

    initialParamsSine = np.array([C, A, omega, p])
    parInit2 = initialParamsSine
    #lBounds = -100*np.ones(len(initParams))
    #uBounds = 100*np.ones(len(initParams))
    nPoints2 = len(xDataSine)
    nPars2 = len(initialParamsSine)

    #Run fit

    outputSine = least_squares(calcChiSqSine, parInit2, args = (xDataSine, yDataSine, xErrorSine, yErrorSine), full_output=True)
    xSine = outputSine[0]
    cov_xSine = outputSine[1]
    
    #print(xGauss)
    #print('\n')
    #print(cov_xGauss)

    # Get least_squares output, stored in array output.x[]
    a = xSine[0]
    b = xSine[1]
    c = xSine[2]
    d = xSine[3]

    # Get errors from our fits using fitStdError(), defined above
    
    pErrors = cov_xSine #fitStdError(output.jac)
    d_a = np.sqrt(pErrors[0][0])
    d_b = np.sqrt(pErrors[1][1])
    d_c = np.sqrt(pErrors[2][2])
    d_d = np.sqrt(pErrors[3][3])
    
    xPlot2 = np.linspace(np.min(xDataSine), np.max(xDataSine), 300)
    fitData2 = fitSine(xSine, xPlot2)

    # Calculate chis**2 per point, summed chi**2 and chi**2/NDF
    chiarr2 = calcChiSqSine(xSine, xDataSine, yDataSine, xErrorSine, yErrorSine)**2
    chisq2 = np.sum(chiarr2)
    NDF2 = nPoints2 - nPars2
    chisqndf2 = chisq2/NDF2
    print("ChiSq = {:5.2e}, ChiSq/NDF = {:5.2f}.".format(chisq2, chisqndf2))

    # Make the plot of the data and the fit
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.errorbar(xDataSine, yDataSine, yerr=yErrorSine, fmt='.', label = 'Simulated Data')
    plt.plot(xPlot2, fitData2, color = 'r', label = "Fit, $\chi^2$/NDF = {0:.2f}".format(chisqndf2))
    plt.title('Momentum Between {0:.0f} MeV and {1:.0f} MeV'.format(p_lim[i][0], p_lim[i][1]))
    plt.xlabel('Time [$\mu$s]')
    plt.ylabel('Average Decay Angle [rad]')
    plt.legend(loc='uppper left', numpoints = 1)
    plt.savefig(outputDir + 'avgAngle_fit.png') 


    # Output fit parameters
    print("Fitted parameters: C= {0:.2f}, A= {1:.4f}, $\omega$= {2:.2f}, p= {3:.2f}".format(a, b, c, d))
    print("Parameter errors: d_C= {0:.2f}, d_B= {1:.4f}, d_$\omega$= {2:.2f}, p= {3:.2f}".format(d_a, d_b, d_c, d_d))

    #Fit for modified sine wave

    #Initial Guesses
    
    C = 0.0
    A = 0.1
    omega = 0.40
    p = 0.0
    B = 0.2

    xDataSine = binCentresSumAnglesTime
    yDataSine = avgAngleTime
    xErrorSine = 0
    yErrorSine = avgAngleTimeErr

    initialParamsSineMod = np.array([C, A, omega, p, B])
    parInitMod = initialParamsSineMod
    #lBounds = -100*np.ones(len(initParams))
    #uBounds = 100*np.ones(len(initParams))
    nPointsMod = len(xDataSine)
    nParsMod = len(initialParamsSineMod)

    #Run fit

    outputSineMod = least_squares(calcChiSqSineMod, parInitMod, args = (xDataSine, yDataSine, xErrorSine, yErrorSine), full_output=True)
    xSineMod = outputSineMod[0]
    cov_xSineMod = outputSineMod[1]
    
    #print(xGauss)
    #print('\n')
    #print(cov_xGauss)

    # Get least_squares output, stored in array output.x[]
    aMod = xSineMod[0]
    bMod = xSineMod[1]
    cMod = xSineMod[2]
    dMod = xSineMod[3]
    eMod = xSineMod[4]

    # Get errors from our fits using fitStdError(), defined above
    
    pErrorsMod = cov_xSineMod #fitStdError(output.jac)
    d_aMod = np.sqrt(pErrorsMod[0][0])
    d_bMod = np.sqrt(pErrorsMod[1][1])
    d_cMod = np.sqrt(pErrorsMod[2][2])
    d_dMod = np.sqrt(pErrorsMod[3][3])
    d_eMod = np.sqrt(pErrorsMod[4][4])
    
    xPlotMod = np.linspace(np.min(xDataSine), np.max(xDataSine), 300)
    fitDataMod = fitSineMod(xSineMod, xPlotMod)

    #print("Max value of mod fit = {0:.4f}".format(max(fitDataMod)))
    fitDataMax.append(max(fitDataMod))

    # Calculate chis**2 per point, summed chi**2 and chi**2/NDF
    chiarrMod = calcChiSqSineMod(xSineMod, xDataSine, yDataSine, xErrorSine, yErrorSine)**2
    chisqMod = np.sum(chiarrMod)
    NDFMod = nPointsMod - nParsMod
    chisqndfMod = chisqMod/NDFMod
    print("ChiSq = {:5.2e}, ChiSq/NDF = {:5.2f}.".format(chisqMod, chisqndfMod))

    # Make the plot of the data and the fit
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.errorbar(xDataSine, yDataSine, yerr=yErrorSine, fmt='.', label = 'Simulated Data')
    plt.plot(xPlotMod, fitDataMod, color = 'r', label = "Fit, $\chi^2$/NDF = {0:.2f}".format(chisqndfMod))
    plt.title('Momentum Between {0:.0f} MeV and {1:.0f} MeV'.format(p_lim[i][0], p_lim[i][1]))
    plt.xlabel('Time [$\mu$s]')
    plt.ylabel('Average Decay Angle [rad]')
    plt.legend(loc='uppper left', numpoints = 1)
    plt.savefig(outputDir + 'avgAngle_fitMod.png') 


    # Output fit parameters
    print("Fitted parameters: C= {0:.2f}, A= {1:.4f}, $\omega$= {2:.2f}, p= {3:.2f}, B = {4:.2f}".format(aMod, bMod, cMod, dMod, eMod))
    print("Parameter errors: d_C= {0:.2f}, d_B= {1:.4f}, d_$\omega$= {2:.2f}, d_p= {3:.2f}, d_B = {4:.2f}".format(d_aMod, d_bMod, d_cMod, d_dMod, d_eMod))

    AVals.append(bMod)
    AValsErr.append(d_bMod)
    BVals.append(eMod)
    BValsErr.append(d_eMod)

    freqMod.append(cMod)
    freqModErr.append(d_cMod)

    #Sine function and modified sine function on same plot

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.errorbar(xDataSine, yDataSine, yerr=yErrorSine, fmt='.', label = 'Simulated Data')
    plt.plot(xPlotMod, fitDataMod, color = 'g', label = "Fit for Modified Sine Function, $\chi^2$/NDF = {0:.2f}".format(chisqndfMod))
    plt.plot(xPlot2, fitData2, color = 'r', label = "Fit for Sine Function, $\chi^2$/NDF = {0:.2f}".format(chisqndf2))
    plt.title('Momentum Between {0:.0f} MeV and {1:.0f} MeV'.format(p_lim[i][0], p_lim[i][1]))
    plt.xlabel('Time [$\mu$s]')
    plt.ylabel('Average Decay Angle [rad]')
    plt.legend(loc='uppper left', numpoints = 1)
    plt.savefig(outputDir + 'avgAngle_fitBoth.png') 

    #Fit for fixed frequency

    #Initial Guesses
    
    C_2 = 0.0
    A_2 = 0.1
    #omega = 0.15
    #p = 0.1

    xDataSine = binCentresSumAnglesTime
    yDataSine = avgAngleTime
    xErrorSine = 0
    yErrorSine = avgAngleTimeErr

    initialParamsSine2 = np.array([C_2, A_2])
    parInit3 = initialParamsSine2
    #lBounds = -100*np.ones(len(initParams))
    #uBounds = 100*np.ones(len(initParams))
    nPoints3 = len(xDataSine)
    nPars3 = len(initialParamsSine2)

    #Run fit

    outputSine2 = least_squares(calcChiSqSine2, parInit3, args = (xDataSine, yDataSine, xErrorSine, yErrorSine), full_output=True)
    xSine2 = outputSine2[0]
    cov_xSine2 = outputSine2[1]
    
    #print(xGauss)
    #print('\n')
    #print(cov_xGauss)

    # Get least_squares output, stored in array output.x[]
    a2 = xSine2[0]
    b2 = xSine2[1]
    #c = xSine[2]
    #d = xSine[3]

    # Get errors from our fits using fitStdError(), defined above
    
    pErrors2 = cov_xSine2 #fitStdError(output.jac)
    d_a2 = np.sqrt(pErrors2[0][0])
    d_b2 = np.sqrt(pErrors2[1][1])
    #d_c = np.sqrt(pErrors[2][2])
    #d_d = np.sqrt(pErrors[3][3])
    
    xPlot3 = np.linspace(np.min(xDataSine), np.max(xDataSine), 300)
    fitData3 = fitSine2(xSine2, xPlot3)

    # Make the plot of the data and the fit
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.errorbar(xDataSine, yDataSine, yerr=yErrorSine, fmt='.', label = 'Simulated Data')
    plt.plot(xPlot3, fitData3, color = 'r', label = "Fit")
    plt.title('Momentum Between {0:.0f} MeV and {1:.0f} MeV'.format(p_lim[i][0], p_lim[i][1]))
    plt.xlabel('Time [$\mu$s]')
    plt.ylabel('Average Decay Angle [rad]')
    plt.legend(loc='uppper left', numpoints = 1)
    plt.savefig(outputDir + 'avgAngle_fitFixed.png') 


    # Output fit parameters
    #print("Fitted parameters: C= {0:.2f}, A= {1:.4f}, $\omega$= {2:.2f}, p= {3:.2f}".format(a, b, c, d))
    #print("Parameter errors: d_C= {0:.2f}, d_B= {1:.4f}, d_$\omega$= {2:.2f}, p= {3:.2f}".format(d_a, d_b, d_c, d_d))

    print("Fitted parameters: C= {0:.2f}, A= {1:.4f}".format(a2, b2))
    print("Parameter errors: d_C= {0:.2f}, d_A= {1:.4f}".format(d_a2, d_b2))


    # Calculate chis**2 per point, summed chi**2 and chi**2/NDF
    chiarr3 = calcChiSqSine2(xSine2, xDataSine, yDataSine, xErrorSine, yErrorSine)**2
    chisq3 = np.sum(chiarr3)
    NDF3 = nPoints3 - nPars3
    chisqndf3 = chisq3/NDF3
    print("ChiSq = {:5.2e}, ChiSq/NDF = {:5.2f}.".format(chisq3, chisqndf3))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.errorbar(binCentresSumAnglesTime, 1/avgAngleTimeErr**2, yerr=0, fmt='.')                                                                                                                                  
    plt.xlabel('Time [$\mu$s]')
    plt.ylabel('Uncertainties')
    plt.savefig(outputDir + 'avgAngles_time_Unc.png')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.errorbar(binCentresSumAnglesTime, 1/errsumAnglesTime**2, yerr=0, fmt='.')                                                                                                                                  
    plt.xlabel('Time [$\mu$s]')
    plt.ylabel('Uncertainties Numerator')
    plt.savefig(outputDir + 'avgAngles_time_UncNum.png')

    print(1/errsumAnglesTime**2)
    print('\n')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.errorbar(binCentresSumAnglesTime, 1/errCounts**2, yerr=0, fmt='.')                                                                                                                                  
    plt.xlabel('Time [$\mu$s]')
    plt.ylabel('Uncertainties Denominator')
    plt.savefig(outputDir + 'avgAngles_time_UncDen.png')

    print(1/errCounts**2)

    amp.append(xSine[1])
    ampError.append(np.sqrt(pErrors[1][1]))

    freq.append(xSine[2])
    freqError.append(np.sqrt(pErrors[2][2]))

    ampFixed.append(xSine2[1])
    ampErrorFixed.append(np.sqrt(pErrors2[1][1]))

    def fitSineB(x, A_3, mu, w, interc):
        '''
        sine wave
        '''
        f= A*np.exp(-(1/2)*((x-mu)/w)**2)+interc 
        return f

    def fitSineDiffB(p, x):
        '''
        Differential of sine fit
        '''
        df= -p[0]*np.exp(-(1/2)*((x-p[1])/p[2])**2)*((x-p[1])/(p[2]**2))
        return df       

    def calcChiSqSineB(p, x, y, xerr, yerr):
        '''
        Error function for fit
        '''
        e = (y - fitSineB(p, x))/(np.sqrt(yerr**2 + fitSineDiffB(p, x)**2*xerr**2))
        return e  


    #Initial Guesses

    A_3 = 0.07
    mu = 14
    w = 2
    interc = 0

    #m = 0
    #intc = 0.0
    #A_3 = 0.07
    #omega = 0.38
    #ph = 0.0

    xDataSine = binCentresSumAnglesTime
    yDataSine = avgAngleTime
    xErrorSine = 0
    yErrorSine = avgAngleTimeErr

    initialParamsSine3 = np.array([A_3, mu, w, interc])
    parInit4 = initialParamsSine3
    #lBounds = -100*np.ones(len(initParams))
    #uBounds = 100*np.ones(len(initParams))
    nPoints4 = len(xDataSine)
    nPars4 = len(initialParamsSine3)

    #Run fit

    popt, pcov = curve_fit(fitSineB, xDataSine, yDataSine, 
                       sigma = yErrorSine, p0 = [A_3, mu, w, interc])
        
    #print(xGauss)
    #print('\n')
    #print(cov_xGauss)

    # Get least_squares output, stored in array output.x[]
    a3 = popt[0]
    b3 = popt[1]
    c3 = popt[2]
    d3 = popt[3]
    #e3 = xSine3[4]


    # Get errors from our fits using fitStdError(), defined above
        
    pErrors3 = pcov #fitStdError(output.jac)
    d_a3 = np.sqrt(pErrors3[0][0])
    d_b3 = np.sqrt(pErrors3[1][1])
    d_c3 = np.sqrt(pErrors3[2][2])
    d_d3 = np.sqrt(pErrors3[3][3])
    #d_e3 = np.sqrt(pErrors3[4][4])
        
    xPlot4 = np.linspace(np.min(xDataSine), np.max(xDataSine), 300)
    #fitData4 = fitSineB(xSine3, xPlot4)

    # Make the plot of the data and the fit
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.errorbar(xDataSine, yDataSine, yerr=yErrorSine, fmt='.', label = 'Simulated Data')
    plt.plot(xPlot4, fitSineB(xPlot4, a3, b3, c3, d3), color = 'r', label = "Fit")
    plt.title('Momentum Between {0:.0f} MeV and {1:.0f} MeV'.format(p_lim[i][0], p_lim[i][1]))
    plt.xlabel('Time [$\mu$s]')
    plt.ylabel('Average Decay Angle [rad]')
    plt.legend(loc='uppper left', numpoints = 1)
    plt.savefig(outputDir + 'avgAngle_fitB.png') 


    # Output fit parameters
    #print("Fitted parameters: C= {0:.2f}, A= {1:.4f}, $\omega$= {2:.2f}, p= {3:.2f}".format(a, b, c, d))
    #print("Parameter errors: d_C= {0:.2f}, d_B= {1:.4f}, d_$\omega$= {2:.2f}, p= {3:.2f}".format(d_a, d_b, d_c, d_d))

    print("Fitted parameters: A= {0:.2f}, mu= {1:.4f}, w = {2:.4f}, c = {3:.4f}".format(a3, b3, c3, d3))
    print("Parameter errors: d_A= {0:.2f}, d_mu= {1:.4f}, d_w = {2:.4f}, d_c = {3:.4f}".format(d_a3, d_b3, d_c3, d_d3))


    # Calculate chis**2 per point, summed chi**2 and chi**2/NDF
    #chiarr4 = calcChiSqSineB(xSine3, xDataSine, yDataSine, xErrorSine, yErrorSine)**2
    #chisq4 = np.sum(chiarr4)
    #NDF4 = nPoints4 - nPars4
    #chisqndf4 = chisq4/NDF4
    #print("ChiSq = {:5.2e}, ChiSq/NDF = {:5.2f}.".format(chisq4, chisqndf4))


for j in range(len(amp)):
    print('Amplitude of fit between {0:.0f} MeV and {1:.0f} MeV = ({2:.3f} +- {3:.3f}) rad'.format(p_lim[j][0], p_lim[j][1], amp[j], ampError[j]))
    print('Frequency of fit between {0:.0f} MeV and {1:.0f} MeV = ({2:.3f} +- {3:.3f}) rad/$\mu$s'.format(p_lim[j][0], p_lim[j][1], freq[j], freqError[j]))
    print('\n')

for ii in range(len(ampFixed)):
    print('Amplitude of fit for fixed frequency between {0:.0f} MeV and {1:.0f} MeV = ({2:.3f} +- {3:.3f}) rad'.format(p_lim[ii][0], p_lim[ii][1], amp[ii], ampError[ii]))
    print('\n')

for k in range(len(freqMod)):
    print('Frequency of modified fit between {0:.0f} MeV and {1:.0f} MeV = ({2:.3f} +- {3:.3f}) rad/$\mu$s'.format(p_lim[k][0], p_lim[k][1], freqMod[k], freqModErr[k]))


print(amp)

amp5 = amp[1:]
ampError5 = ampError[1:]
freq5 = freq[1:]
freqError5 = freqError[1:]

#Original fit frequencies

countsAmp, binEdgesAmp = np.histogram(tot_posIniMom_array, bins = 5, range=(0, p_max))
binCentresAmp = (binEdgesAmp[:-1] + binEdgesAmp[1:]) / 2
binCentresAmp5 = binCentresAmp[1:]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.errorbar(binCentresAmp, amp5, yerr=ampError5, fmt='.')                                                                                                                                 
plt.xlabel('Positron Momentum [Mev/c]')
plt.ylabel('Amplitude of Polarisation Vector Oscillation [rad]')
plt.savefig(outputDirTop + 'amplitude_mom.png')

countsFreq, binEdgesFreq = np.histogram(tot_posIniMom_array, bins = 5, range=(0, p_max))
binCentresFreq = (binEdgesFreq[:-1] + binEdgesFreq[1:]) / 2
binCentresFreq5 = binCentresFreq[1:]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.errorbar(binCentresFreq, freq5, yerr=freqError5, fmt='.')                                                                                                                                 
plt.xlabel('Positron Momentum [Mev/c]')
plt.ylabel('Frequency of Polarisation Vector Oscillation [rad/$\mu$s]')
plt.savefig(outputDirTop + 'freq_mom.png')

#Modified fit freq

countsFreqMod, binEdgesFreqMod = np.histogram(tot_posIniMom_array, bins = 5, range=(0, p_max))
binCentresFreqMod = (binEdgesFreqMod[:-1] + binEdgesFreqMod[1:]) / 2
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.errorbar(binCentresFreqMod, freqMod[1:], yerr=freqModErr[1:], fmt='.')                                                                                                                                 
plt.xlabel('Positron Momentum [Mev/c]')
plt.ylabel('Frequency of Polarisation Vector Oscillation [rad/$\mu$s]')
plt.savefig(outputDirTop + 'freqMod_mom.png')

#Freq Fit

freqModArr = np.array(freqMod[1:], dtype='float32')
freqModErrorArr = np.array(freqModErr[1:], dtype='float32')

def fitFunc(p, x):
    '''
    Fit function
    '''
    f = p[0]*x +p[1]
    return f

def fitFuncDiff(p, x):
    '''
    Differential of fit function
    '''
    df= p[0]
    return df

def calcChiSq(p, x, y, xerr, yerr):
    '''
    Error function for fit
    '''
    e = (y - fitFunc(p, x))/(np.sqrt(yerr**2 + fitFuncDiff(p, x)**2*xerr**2))
    return e

xdata = binCentresFreqMod
ydata = freqModArr
xerror = 0
#yerror = errCounts*1/countsTime
yerror = freqModErrorArr

#print(counts)
#print(err)
#print(yerror)

initParams = [1, 1]
pInit = initParams
lBounds = -100*np.ones(len(initParams))
uBounds = 100*np.ones(len(initParams))
nPoints = len(xdata)
nPars = len(initParams)

# Run fit
#output = least_squares(calcChiSq, pInit, args = (xdata, ydata, xerror, yerror))
outputFreqMod = least_squares(calcChiSq, pInit, args = (xdata, ydata, xerror, yerror), full_output=True)
xFreqMod = outputFreqMod[0]
cov_xFreqMod = outputFreqMod[1]                

#print(x)
#print(cov_x)
#print(len(output))

# Get least_squares output, stored in array output.x[]
mFreqMod = xFreqMod[0]
cFreqMod = xFreqMod[1]


# Get errors from our fits using fitStdError(), defined above
pErrorsFreqMod = cov_xFreqMod #fitStdError(output.jac)
d_mFreqMod = np.sqrt(pErrorsFreqMod[0][0])
d_cFreqMod = np.sqrt(pErrorsFreqMod[1][1])

# Calculate fitted y-values using our fit parameters and the original fit function
xPlotFreqMod = np.linspace(np.min(xdata), np.max(xdata), 300)
fitDataFreqMod = fitFunc(xFreqMod, xPlotFreqMod)   

# Calculate chis**2 per point, summed chi**2 and chi**2/NDF
chiarrFreqMod = calcChiSq(xFreqMod, xdata, ydata, xerr = xerror, yerr = yerror)**2
chisqFreqMod = np.sum(chiarrFreqMod)
NDF = nPoints - nPars
chisqndfFreqMod = chisqFreqMod/NDF
print("ChiSq = {:5.2e}, ChiSq/NDF = {:5.2f}.".format(chisqFreqMod, chisqndfFreqMod)) 

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#ax.set_yscale("log")
plt.errorbar(xdata, ydata, yerr=yerror, fmt='.', label = 'Simulated Data')
plt.plot(xPlotFreqMod, fitDataFreqMod, color = 'r', label = "Fit, $\chi^2$/NDF = {0:.2f}".format(chisqndfFreqMod))
plt.xlabel('Positron Momentum [Mev/c]')
plt.ylabel('Frequency of Polarisation Vector Oscillation [rad/$\mu$s]')
plt.legend(numpoints=1)
plt.savefig(outputDirTop + 'freqMod_momFit.png')     

# Output fit parameters
print("Fitted parameters: m = {0:.4f}, c = {1:.6f}".format(mFreqMod, cFreqMod))
print("Parameter errors: d_m ={0:.2f}, d_c = {1:.8f}".format(d_mFreqMod, d_cFreqMod))


B = 3 #tesla                                                                                                                                                                                                              
amu = 0.00116591
c = 3E8
beta = 0.77 # 0.82 to give gamma factor from note                                                                                                                                                                         
gamma = 1/np.sqrt(1-beta**2)

print("gamma factor:", gamma)

Ef = amu*B*c*beta* (gamma**2)
print(Ef , "V/m = ", Ef /1000, "kV/m = ", Ef / (1000*100), "kV/cm")

p1 = beta*B
p2 = Ef/c

print(p1, p2)

hbar = 1.054E-34
e = 1.6E-19 #C                                                                                                                                                                                                            
m = 1.883E-28 #kg                                                                                                                                                                                                         

#input value from sim                                                                                                                                                                                                     
#we = cFreqMod*10**6 #0.1896E6 #2*np.pi / (40E-6) #-s   
we = cFreqMod*1000000
weErr = d_cFreqMod*10**6

print("omega_e:", we, "rads/s, giving T = ", 2*np.pi / we, "s")

nu = (2*m*we) / (e*(p1+p2))
nuErr = (2*m/e*(p1+p2))*weErr

dmu = we * hbar / (2*(c * (p1+p2))) #C.m
dmuErr = weErr * hbar / (2*(c * (p1+p2))) #C.m

dmuecm = dmu*100/e
dmuecmErr = dmuErr*100/e

#print("nu: ", nu,  " dmu:", dmu, "C.m", dmu*100, "C.cm", dmu*100/e, "e.cm")
print("Muon EDM Value = ({0:.4e} +- {1:.4e}) e.cm".format(dmuecm, dmuecmErr))

P_0 = 0.95
APol = 0.3

sensitivity = (amu*hbar*gammaFactor)/(2*P_0*Ef*np.sqrt(n_events)*muonLifetimeLit*APol)
sensECM = sensitivity*100/e

print("Sensitivity of muon EDM =",sensECM)

SES = (amu*hbar*gammaFactor)/(2*P_0*Ef*muonLifetimeLit*APol)
SESecm = SES*100/e

print("Single event sensitivity of muon EDM =", SESecm)

countsB, binEdgesB = np.histogram(tot_posIniMom_array, bins = 5, range=(0, p_max))
binCentresB = (binEdgesB[:-1] + binEdgesB[1:]) / 2
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.errorbar(binCentresB, BVals[1:], yerr=BValsErr[1:], fmt='.')                                                                                                                                 
plt.xlabel('Positron Momentum [Mev/c]')
plt.ylabel('Amplitude of Cosine Function [rad]')
plt.savefig(outputDirTop + 'BVals_mom.png')

countsA, binEdgesA = np.histogram(tot_posIniMom_array, bins = 5, range=(0, p_max))
binCentresA = (binEdgesA[:-1] + binEdgesA[1:]) / 2
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.errorbar(binCentresA, AVals[1:], yerr=AValsErr[1:], fmt='.')                                                                                                                                 
plt.xlabel('Positron Momentum [Mev/c]')
plt.ylabel('Amplitude of Sine Function [rad]')
plt.savefig(outputDirTop + 'AVals_mom.png')

countsBTime, binEdgesBTime = np.histogram(tot_muDecayTime, bins = 5)
binCentresBTime = (binEdgesBTime[:-1] + binEdgesBTime[1:]) / 2
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.errorbar(binCentresBTime, BVals[1:], yerr=BValsErr[1:], fmt='.')                                                                                                                                 
plt.xlabel('Time ($\mu$s)')
plt.ylabel('Amplitude of Cosine Function [rad]')
plt.savefig(outputDirTop + 'BVals_time.png')

countsATime, binEdgesATime = np.histogram(tot_muDecayTime, bins = 5)
binCentresATime = (binEdgesATime[:-1] + binEdgesATime[1:]) / 2
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.errorbar(binCentresATime, AVals[1:], yerr=AValsErr[1:], fmt='.')                                                                                                                                 
plt.xlabel('Time ($\mu$s)')
plt.ylabel('Amplitude of Sine Function [rad]')
plt.savefig(outputDirTop + 'AVals_Time.png')

countsUncTime, binEdgesUncTime = np.histogram(tot_posIniMom_array, bins = 5, range = (0, p_max))
binCentresUncTime = (binEdgesUncTime[:-1] + binEdgesUncTime[1:]) / 2
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.errorbar(binCentresUncTime, BValsErr[1:], yerr=0, fmt='.')                                                                                                                                 
plt.xlabel('Time ($\mu$s)')
plt.ylabel('Uncertainties [rad]')
plt.savefig(outputDirTop + 'BVals_Time_Unc.png')

counts, binEdges = np.histogram(tot_posIniMom_array, bins = nBins, range = (0, p_max))
binCentres = (binEdges[:-1] + binEdges[1:]) / 2
err = np.sqrt(counts)
countsArr = np.repeat(max(counts), nBins)
#print(countsArr)
countsNorm = counts/(countsArr)
#print(counts)
#print(max(counts))
#print(countsNorm)
errNorm = err/max(counts)
plt.errorbar(binCentres, counts, yerr=err, fmt='.')
plt.xlabel("Positron Momentum [MeV/c]")
plt.ylabel("Normalised Number of Events")
plt.axvline(x = 0, linestyle = '--', color = 'g')
plt.axvline(x = 25, linestyle = '--', color = 'g')
plt.axvline(x = 50, linestyle = '--', color = 'g')
plt.axvline(x = 75, linestyle = '--', color = 'g')
plt.axvline(x = 100, linestyle = '--', color = 'g')
plt.axvline(x = 125, linestyle = '--', color = 'g')
plt.savefig(outputDirTop + 'positron_mom_cuts.png') 


print("A Vals:", AVals)
print("A Vals Errors:", AValsErr)

print()

print("Amp sine fit:", amp)
print("Amp sine fit errors:", ampError)

print("Freq sine fit:", freq)
print("Freq sine fit errors:", freqError)

print()

print("Freq mod fit:", freqMod)
print("Freq mod fit errors:", freqModErr)