#%%
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy import integrate, stats
import matplotlib.pyplot as plt
from matplotlib import gridspec, rcParams, rcParamsDefault
import json
import csv
import pandas as pd
import seaborn as sns
from scipy.stats import linregress, pearsonr, mode
from functions_phone import *
import random
#%%
##############################
###    FACEBOOK DATASET    ###
##############################
path = '/home/ignacio/Dropbox/DOCTORADO/DUNDIG/Data/FB'
with open(path + '/mago.csv','r') as f:
    temp = csv.reader(f)
    raw_data = [row for row in temp]#%%

#%%
Eta = []
samples = len(raw_data)
examples_inverted = []
examples_standard = []
all_cases = []
standard = []
inverted = []
for row in raw_data[:samples]:
    
    x = np.array([float(x) for x in row if x != ''])
    smax = max(x)
    smin = min(x)
    delta_s = smax - smin
    L = len(x)
    S = sum(x)
    g_exp = (smax - S/L) / delta_s
    L1 = g_exp*L
    
    # Total number of links FILTER VERY LOW INTERACTIONS
    if L > 0:
        try:
            eta = fsolve((lambda y: g(y) - g_exp),x0=1.,xtol=1e-6)[0]
            Eta.append(eta)
            all_cases.append(x)

            if eta < -3 and eta >-9:
                examples_inverted.append(x)
            if eta > 3 and eta <9:
                examples_standard.append(x)
            if eta < 0.:
                inverted.append(x)
            if eta > 0.:
                standard.append(x)
        except:
            pass
        
#%%##########################################
####         FIT ALL INVERTED            ####
############################################
# SELECT ONLY THOSE WHOSE CI DOES NOT INCLUDE 0
            
totally_inverted = []
for x in inverted:
    x =sorted(x,reverse=True)
    smax = max(x)
    smin = min(x)
    delta_s = smax - smin
    L = len(x)
    S = sum(x)
    g_exp = (smax - S/L) / delta_s
    L1 = g_exp*L
    eta = fsolve((lambda y: g(y) - g_exp),x0=1.,xtol=1e-6)[0]
    t1, t2 = eta_confidence_interval(L1, L, d=0.05, initial=eta,tol=1e-6)
    if t1 < 0 and t2 < 0:
        totally_inverted.append(x)
#%%
print 'percentage of true inverted =', len(totally_inverted)*100 / float(len(all_cases))
print ' percentage of inverted =', len(inverted)*100 / float(len(all_cases))
#%%##########################################
####         FIT ALL STANDARD            ####
############################################
# SELECT ONLY THOSE WHOSE CI DOES NOT INCLUDE 0
            
totally_standard = []
for x in standard:
    x =sorted(x,reverse=True)
    smax = max(x)
    smin = min(x)
    delta_s = smax - smin
    L = len(x)
    S = sum(x)
    g_exp = (smax - S/L) / delta_s
    L1 = g_exp*L
    eta = fsolve((lambda y: g(y) - g_exp),x0=1.,xtol=1e-6)[0]
    t1, t2 = eta_confidence_interval(L1, L, d=0.05, initial=eta,tol=1e-6)
    if t1 > 0 and t2 > 0:
        totally_standard.append(x)
#%%

with open('totally_standard_facebook.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(totally_standard)
#%%##############################
###    FACEBOOK FIGURE        ###
#################################

##### http://matplotlib.org/users/customizing.html
    #https://github.com/jbmouret/matplotlib_for_papers#subplots
    #https://matplotlib.org/users/dflt_style_changes.html
    #https://matplotlib.org/1.5.1/users/customizing.html
def cm_to_inch(x):
    return x*0.393701
    
rcParams.update(rcParamsDefault)
plt.style.use(['ggplot'])
params = {
    'font.family' : "Times New Roman",
    'lines.markersize' :5,  #scatter points
    'figure.facecolor' : 'w', 
    'lines.linewidth' : 1.6,
    'axes.titlesize' : 16,
    'axes.labelsize': 10,
    'axes.labelpad' : 3,
    'font.size': 12, # for the a,b,c...
    'legend.fontsize': 50,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'text.usetex': False,
    'figure.figsize': [cm_to_inch(11.4), cm_to_inch(6.8)] #[w,h] [width=11.4cm,height=6.8cm]
   }
rotate = 20
markersize=5
rcParams.update(params)


#[u'seaborn-darkgrid', u'seaborn-notebook', u'classic', u'seaborn-ticks', u'grayscale', 
#u'bmh', u'seaborn-talk', u'dark_background', u'ggplot', u'fivethirtyeight', u'seaborn-colorblind',
# u'seaborn-deep', u'seaborn-whitegrid', u'seaborn-bright', u'seaborn-poster', u'seaborn-muted', 
#u'seaborn-paper', u'seaborn-white', u'seaborn-pastel', u'seaborn-dark', u'seaborn-dark-palette']
# Cool Styles
# seaborn-darkgrid, seaborn-talk, ggplot, fivethirtyeight, seaborn-colorblind
# seaborn-deep
# Plot figure with subplots of different sizes




np.random.seed(1)
fig = plt.figure(1)
grid = (2,3)
gridspec.GridSpec(2,3)
# LARGE FIGURE: HISTOGRAM
plt.subplot2grid(grid, (0,0), colspan=2, rowspan=2)
not_extreme = [x for x in Eta if abs(x)<np.inf] #DOES NOTHING
#log_ = [np.log(x) for x in not_extreme if x > 0.]
#print np.mean(not_extreme), np.median(not_extreme),stats.mode(not_extreme)
plt.hist(not_extreme,bins='auto',normed=True,color='#3b5998',alpha=0.3, log=False)
#plt.title('Facebook Data')
plt.xlabel(r'$\eta$')
plt.ylabel('frequency')
#plt.savefig('facebook_good_t_reescaled.pdf')
plt.axvline(x=0,ymin=0,ymax=8,linestyle ='--', color='r')
plt.xlim((-10,30))

# small subplot 1 STANDARD
    # data

#case = np.random.randint(0,len(examples_standard))
#case = np.random.randint(0,len(standard))
case = np.random.randint(0,len(totally_standard))
x = totally_standard[case]

#case = 54
#x = examples_standard[case]
#x = standard[case]

x =sorted(x,reverse=True)
smax = max(x)
smin = min(x)
delta_s = smax - smin
L = len(x)
S = sum(x)
g_exp = (smax - S/L) / delta_s
L1 = g_exp*L
eta = fsolve((lambda y: g(y) - g_exp),x0=1.,xtol=1e-6)[0]
t1, t2 = eta_confidence_interval(L1, L, d=0.05, initial=eta,tol=1e-6)
ex_standard = (eta, t1, t2, L)
X = np.linspace(smin,smax,num=1000)
Y = np.array([function_to_fit(t,eta,smax,smin) for t in X])      # estimated eta
curve1 = [function_to_fit(t,t1,smax,smin) for t in X]            # lower bound of CI
curve2 = [function_to_fit(t,t2,smax,smin) for t in X]            # upper bound of CI
x = [smax - t + smin for t in x] #uncomment to plot as in discrete case
X= X[::-1] 
    # Plot
plt.subplot2grid(grid, (0,2))
plt.scatter(x, y(x,True),marker='o',color='r',s=markersize, alpha=0.4)                                        # empirical values
plt.plot(X,Y,linestyle='--',color='b')                              # fitted 
plt.xticks([smin,smax], [0,1])
plt.fill_between(X, curve1, curve2, alpha = 0.25, color='dodgerblue')         # confidence interval
plt.ylabel(r'$\chi(t)$')
plt.xlabel(r'$t$',labelpad=-3)
plt.ylim((0,1.1))
plt.xlim((smin - (smax-smin) /30.,smax+(smax-smin) /30.))

# small subplot 2 INVERSE
    # data
#case = np.random.randint(0,len(examples_inverted))
#x = examples_inverted[case]
case = np.random.randint(0,len(totally_inverted))
x = totally_inverted[case]
x =sorted(x,reverse=True)
smax = max(x)
smin = min(x)
delta_s = smax - smin
L = len(x)
S = sum(x)
g_exp = (smax - S/L) / delta_s
L1 = g_exp*L
eta = fsolve((lambda y: g(y) - g_exp),x0=1.,xtol=1e-6)[0]
t1, t2 = eta_confidence_interval(L1, L, d=0.05, initial=eta,tol=1e-6)
ex_inverted = (eta, t1, t2, L)

X = np.linspace(smin,smax,num=1000)
Y = np.array([function_to_fit(t,eta,smax,smin) for t in X])      # estimated eta
curve1 = [function_to_fit(t,t1,smax,smin) for t in X]            # lower bound of CI
curve2 = [function_to_fit(t,t2,smax,smin) for t in X]            # upper bound of CI
x = [smax - t + smin for t in x] #uncomment to plot as in discrete case
X= X[::-1] 
    # Plot
plt.subplot2grid(grid, (1,2))
plt.scatter(x, y(x,True),marker='o',color='r',s=markersize, alpha=0.4)                                        # empirical values
plt.plot(X,Y,linestyle='--',color='b')                              # fitted 
plt.xticks([smin,smax], [0,1])
plt.fill_between(X, curve1, curve2, alpha = 0.25, color='dodgerblue')         # confidence interval
plt.ylabel(r'$\chi(t)$')
plt.xlabel(r'$t$',labelpad=-3)
plt.ylim((0,1.1))
plt.xlim((smin - (smax-smin) /30.,smax+(smax-smin) /30.))

fig.text(0.04, 0.94, "a", weight="bold", horizontalalignment='left', verticalalignment='bottom')
fig.text(0.70, 0.94, "b", weight="bold", horizontalalignment='left', verticalalignment='bottom')
fig.text(0.70, 0.51, "c", weight="bold", horizontalalignment='left', verticalalignment='bottom')


plt.tight_layout()
plt.savefig('facebook_thesis.pdf',dpi=800)

plt.show()
#%%
print np.mean(Eta)

print np.median(Eta)

print np.std(Eta)

max_bin = max(np.histogram(Eta,bins=1000)[0])
max_bin_index = list(np.histogram(Eta,bins=1000)[0]).index(max_bin)
mode = np.histogram(Eta,bins=1000)[1][max_bin_index]
print mode
#%%
print sum([1  for x in Eta if x<0])




#%%
############################################
####         COMPLETE PLOTS (Apnd)      ####
############################################
def cm_to_inch(x):
    return x*0.393701
    
rcParams.update(rcParamsDefault)
plt.style.use(['ggplot'])
params = {
    'font.family' : "Times New Roman",
    'lines.markersize' :5,  #scatter points
    'figure.facecolor' : 'w', 
    'lines.linewidth' : 1.6,
    'axes.titlesize' : 18,
    'axes.labelsize': 40,
    'axes.labelpad' : 2,
    'text.fontsize': 10, # for the a,b,c...
    'legend.fontsize': 50,
    'xtick.labelsize': 10,
    'ytick.labelsize': 6,
    'text.usetex': False,
    'figure.figsize': [cm_to_inch(17.8), cm_to_inch(17.8 / 1.1)] #[w,h] [width=11.4cm,height=6.8cm]
   }
plot_as_PNAS=True
markersize=13
########################################
####        PLOTS SI                ####
########################################
m=3
n=8
fig, axes = plt.subplots(m,n,sharey=True,figsize=(11.7,8.3))
rows = range(m)
cols = range(n)
np.random.seed(1)

population = all_cases#inverted#standard#all_cases
sample = np.random.permutation(population)[:24]
case = 0
for i in rows:
    for j in cols:
        x = sample[case]
        x =sorted(x,reverse=True)
        smax = max(x)
        smin = min(x)
        delta_s = smax - smin
        L = len(x)
        S = sum(x)
        g_exp = (smax - S/L) / delta_s
        L1 = g_exp*L
        eta = fsolve((lambda y: g(y) - g_exp),x0=1.,xtol=1e-6)[0]
        t1, t2 = eta_confidence_interval(L1, L, d=0.05, initial=eta,tol=1e-6)

        X = np.linspace(smin,smax,num=1000)
        Y = np.array([function_to_fit(t,eta,smax,smin) for t in X])      # estimated eta
        curve1 = [function_to_fit(t,t1,smax,smin) for t in X]            # lower bound of CI
        curve2 = [function_to_fit(t,t2,smax,smin) for t in X]            # upper bound of CI
        #X= X[::-1] #uncomment to plot as in discrete case
        if plot_as_PNAS:
            x = [smax - t + smin for t in x] #uncomment to plot as in discrete case
            X= X[::-1] 
        axes[i][j].scatter(x, y(x,plot_as_PNAS),marker='o',color='r',s=markersize, alpha=0.8)    # empirical values
        axes[i][j].plot(X,Y,linestyle='--',color='b',linewidth=4)                              # fitted 
        axes[i][j].set_xticks([smin,smax])
        axes[i][j].set_xticklabels([0,1],size=15)
        if j == 0:
            axes[i][j].set_ylabel(r'$\chi(t)$',size=20)
#        axes[i][j].set_title(dataI1.index[case])
        axes[i][j].set_ylim(-0.01,1.01)
        axes[i][j].fill_between(X, curve1, curve2, alpha = 0.25)         # confidence interval
        axes[i][j].set_xlabel(r'$t$',labelpad=-12,size=18)
        case += 1

fig.tight_layout()
#fig.suptitle('Time 1')
plt.savefig('FB_Appendix.pdf',dpi=600)
plt.show()


#%%
##############################
###     SOCIOPATTERNS      ###
##############################

#Time is measured in seconds since 8am on Jun 29th 2009 (UNIX ctime 1246255200).

 
path = '/home/ignacio/Dropbox/DOCTORADO/DUNDIG/Data/SocioPatterns/HyperText2009/'
with open(path+'ht09_contact_intervals.json', 'r') as fp:
    data = json.load(fp)
#%%

clean_data = total_interaction(data)#.values()

appendix_fig = False
if appendix_fig:
    smin_smax = smin_smax(data) #UNCOMMENT FOR FIG IN APPENDIX
#%%
Eta = []
examples_inverted = []
examples_standard = []
standard = []
for ego,row in clean_data.iteritems():
    
    x = np.array([float(x) for x in row if x != ''])
    smax = max(x)
    smin = min(x) 
    if appendix_fig:
        smin, smax = smin_smax[ego] #UNCOMMENT FOR FIG IN APPENDIX
    delta_s = smax - smin
    L = len(x)
        # Total number of links FILTER VERY LOW INTERACTIONS
    if (L >5):
        standard.append(ego) # they are all cases indeed
        S = sum(x)
        g_exp = (smax - S/L) / delta_s
        L1 = g_exp*L
        try:
            eta = fsolve((lambda y: g(y) - g_exp),x0=1.,xtol=1e-6)[0]
            Eta.append(eta)
            if eta < -3 and eta >-9:
                examples_inverted.append(ego)
            if eta > 3 and eta < 9:
                examples_standard.append(ego)
        except:
            pass
#%%
# SELECT ONLY THOSE WHOSE CI DOES NOT INCLUDE 0
            
totally_standard = []
for x in standard:
    x =sorted(clean_data[x],reverse=True)
   
    smax = max(x)
    smin = min(x)
    delta_s = smax - smin
    L = len(x)
    S = float(sum(x))
    g_exp = (smax - S/L) / delta_s
    L1 = g_exp*L
    eta = fsolve((lambda y: g(y) - g_exp),x0=1.,xtol=1e-6)[0]
    t1, t2 = eta_confidence_interval(L1, L, d=0.05, initial=eta,tol=1e-6)
    if t1 > 0 and t2 > 0:
        print eta
        totally_standard.append(x)
#%%################################
###    SOCIOPATTERNS FIGURE     ###
###################################

##### http://matplotlib.org/users/customizing.html
    #https://github.com/jbmouret/matplotlib_for_papers#subplots
    #https://matplotlib.org/users/dflt_style_changes.html
    #https://matplotlib.org/1.5.1/users/customizing.html
def cm_to_inch(x):
    return x*0.393701
    
rcParams.update(rcParamsDefault)
plt.style.use(['ggplot'])
params = {
    'font.family' : "Times New Roman",
    'lines.markersize' :5,  #scatter points
    'figure.facecolor' : 'w', 
    'lines.linewidth' : 1.6,
    'axes.titlesize' : 16,
    'axes.labelsize': 10,
    'axes.labelpad' : 3,
    'text.fontsize': 12, # for the a,b,c...
    'legend.fontsize': 50,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'text.usetex': False,
    'figure.figsize': [cm_to_inch(11.4), cm_to_inch(5.7)] #[w,h] [width=11.4cm,height=6.8cm]
   }
rotate = 20
markersize=7
rcParams.update(params)


#[u'seaborn-darkgrid', u'seaborn-notebook', u'classic', u'seaborn-ticks', u'grayscale', 
#u'bmh', u'seaborn-talk', u'dark_background', u'ggplot', u'fivethirtyeight', u'seaborn-colorblind',
# u'seaborn-deep', u'seaborn-whitegrid', u'seaborn-bright', u'seaborn-poster', u'seaborn-muted', 
#u'seaborn-paper', u'seaborn-white', u'seaborn-pastel', u'seaborn-dark', u'seaborn-dark-palette']
# Cool Styles
# seaborn-darkgrid, seaborn-talk, ggplot, fivethirtyeight, seaborn-colorblind
# seaborn-deep
# Plot figure with subplots of different sizes


fig,axes = plt.subplots(1,2)
#np.random.seed(1)
# HISTOGRAM 
Eta = [e for e in Eta if e < 100]
#nbins='auto'
nbins=15
axes[0].hist(Eta,bins=nbins,normed=True,color='#FF5E05',alpha=0.6, log=False)
axes[0].set_xlabel(r'$\eta$')
axes[0].set_ylabel('frequency')
axes[0].axvline(x=0,ymin=0,ymax=8,linestyle ='--', color='r')
axes[0].set_xlim((-1,50))


# EXAMPLE
    # data
#case = np.random.randint(0,len(examples_standard)) 
#case = np.random.randint(0,len(totally_standard)) 

#if appendix_fig:
#    case = 12
#ego = examples_standard[case]
#ego = standard[case]

#x = sorted(clean_data[ego],reverse=True)
#x = totally_standard[case]
np.random.seed(8)

x = np.random.choice(totally_standard)
smax = max(x)
smin = min(x)
if appendix_fig:
    smin, smax = smin_smax[ego] #UNCOMMENT FOR FIG IN APPENDIX
delta_s = smax - smin
L = len(x)
S = float(sum(x))
g_exp = (smax - S/L) / delta_s
L1 = g_exp*L
eta = fsolve((lambda y: g(y) - g_exp),x0=1.,xtol=1e-6)[0]
t1, t2 = eta_confidence_interval(L1, L, d=0.05, initial=eta,tol=1e-6)
ex_standard = (eta, t1, t2, L)

X = np.linspace(smin,smax,num=1000)
Y = np.array([function_to_fit(t,eta,smax,smin) for t in X])      # estimated eta
curve1 = [function_to_fit(t,t1,smax,smin) for t in X]            # lower bound of CI
curve2 = [function_to_fit(t,t2,smax,smin) for t in X]            # upper bound of CI
x = [smax - t + smin for t in x] #uncomment to plot as in discrete case
X= X[::-1] 
    # Plot
axes[1].scatter(x, y(x,True),marker='o',color='r',s=markersize, alpha=0.6)                                        # empirical values
axes[1].plot(X,Y,linestyle='--',color='b')                              # fitted 
axes[1].set_xticks([smin,smax])
axes[1].set_xticklabels([0,1])

axes[1].fill_between(X, curve1, curve2, alpha = 0.25, color='dodgerblue')         # confidence interval
axes[1].set_ylabel(r'$\chi(t)$')
axes[1].set_xlabel(r'$t$')
axes[1].set_ylim((0,1.1))
axes[1].set_xlim((smin - (smax-smin) /30.,smax+(smax-smin) /30.))

fig.text(0.04, 0.91, "a", weight="bold", horizontalalignment='left', verticalalignment='bottom')
fig.text(0.55, 0.91, "b", weight="bold", horizontalalignment='left', verticalalignment='bottom')
#fig.text(0.70, 0.51, "c", weight="bold", horizontalalignment='left', verticalalignment='bottom')


plt.tight_layout()
if appendix_fig:
    plt.savefig('sociopatterns_thesis_appendix.pdf',dpi=600)
else:
    plt.savefig('sociopatterns_thesis.pdf',dpi=600)
#plt.savefig('sociopatterns_thesis.pdf',dpi=600)   
plt.show()
#%%
print ex_standard
print np.mean(Eta)

print np.median(Eta)
print len(Eta)
print np.std(Eta)


max_bin = max(np.histogram(Eta,bins=nbins)[0])
max_bin_index = list(np.histogram(Eta,bins=nbins)[0]).index(max_bin)
mode = np.histogram(Eta,bins=nbins)[1][max_bin_index]
print mode
#%%
############################################
####         COMPLETE PLOTS (Apnd)      ####
############################################
def cm_to_inch(x):
    return x*0.393701
    
rcParams.update(rcParamsDefault)
plt.style.use(['ggplot'])
params = {
    'font.family' : "Times New Roman",
    'lines.markersize' :5,  #scatter points
    'figure.facecolor' : 'w', 
    'lines.linewidth' : 1.6,
    'axes.titlesize' : 18,
    'axes.labelsize': 40,
    'axes.labelpad' : 2,
    'text.fontsize': 10, # for the a,b,c...
    'legend.fontsize': 50,
    'xtick.labelsize': 10,
    'ytick.labelsize': 6,
    'text.usetex': False,
    'figure.figsize': [cm_to_inch(17.8), cm_to_inch(17.8 / 1.1)] #[w,h] [width=11.4cm,height=6.8cm]
   }
plot_as_PNAS=True
markersize=13
########################################
####        PLOTS SI                ####
########################################
m=3
n=8
fig, axes = plt.subplots(m,n,sharey=True,figsize=(11.7,8.3))
rows = range(m)
cols = range(n)
np.random.seed(8)

population = [clean_data[ego] for ego in standard]#clean_data.values()#all_cases#inverted#standard#all_cases
sample = np.random.permutation(population)[:24]
case = 0
for i in rows:
    for j in cols:
        x = sample[case]
        x =sorted(x,reverse=True)
        smax = max(x)
        smin = min(x)
        delta_s = smax - smin
        L = len(x)
        S = float(sum(x))
        g_exp = (smax - S/L) / delta_s
        L1 = g_exp*L
        eta = fsolve((lambda y: g(y) - g_exp),x0=1.,xtol=1e-6)[0]
        t1, t2 = eta_confidence_interval(L1, L, d=0.05, initial=eta,tol=1e-6)

        X = np.linspace(smin,smax,num=1000)
        Y = np.array([function_to_fit(t,eta,smax,smin) for t in X])      # estimated eta
        curve1 = [function_to_fit(t,t1,smax,smin) for t in X]            # lower bound of CI
        curve2 = [function_to_fit(t,t2,smax,smin) for t in X]            # upper bound of CI
        #X= X[::-1] #uncomment to plot as in discrete case
        if plot_as_PNAS:
            x = [smax - t + smin for t in x] #uncomment to plot as in discrete case
            X= X[::-1] 
        axes[i][j].scatter(x, y(x,plot_as_PNAS),marker='o',color='r',s=markersize, alpha=0.8)    # empirical values
        axes[i][j].plot(X,Y,linestyle='--',color='b',linewidth=4)                              # fitted 
        axes[i][j].set_xticks([smin,smax])
        axes[i][j].set_xticklabels([0,1],size=15)
        if j == 0:
            axes[i][j].set_ylabel(r'$\chi(t)$',size=20)
#        axes[i][j].set_title(dataI1.index[case])
        axes[i][j].set_ylim(-0.01,1.01)
        axes[i][j].fill_between(X, curve1, curve2, alpha = 0.25)         # confidence interval
        axes[i][j].set_xlabel(r'$t$',labelpad=-12,size=18)
        case += 1

fig.tight_layout()
#fig.suptitle('Time 1')
plt.savefig('SP_Appendix.pdf',dpi=600)
plt.show()


