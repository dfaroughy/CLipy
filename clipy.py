from __future__ import print_function
from scipy import stats, optimize, integrate, special
from scipy import __version__ as scipy_version
from numpy  import sqrt, exp, log, sign, array, ndarray
from scipy.stats import poisson, norm
from iminuit import minimize
import pymc3 as pm
import numpy as np
import math
import copy
import sys
import random


'''
------------------------------------------------------------------------------------
-------------------------  CLIPY: COLLIDER LIMITS IN PYTHON  -----------------------
------------------------------------------------------------------------------------
* Computes p-values, CLs, etc using likelihood ratios based on CMS-NOTE-2017/001.

* Distributions of the test statistics are extracted using MC pseudo-experiments, 
  or asymptotic formulae arxiv:1007.1727.

* Minimization performed in MINUIT via iminuit.

* MC toys are egenrated using PYMC3.
------------------------------------------------------------------------------------

'''

class data__model:
    def __init__(self,nobs,s,b,s_err,b_err):
        self.nbins    = len(nobs)
        self.obs    = nobs 
        self.signal = s
        self.backgr = b
        if s_err!=None:
            self.s_err = s_err
       	else:
       	    self.s_err = [0.]*self.nbins
        if b_err!=None:
            self.b_err = b_err
        else:
            self.b_err = [0.]*self.nbins

    def toys(self,N_toys=1000,cores=4,chains=1):
        pseudo_experiments=[]
        for i in range(self.nbins):
            with pm.Model():
            	pm.Poisson('obs_toy', mu=self.obs[i])
            	# pm.Poisson('exp_toy', mu=self.signal[i]+self.backgr[i])
                pm.Poisson('s_toy',   mu=self.signal[i])
                pm.Poisson('b_toy',   mu=self.backgr[i])
            	pm.Normal('s_err_toy', mu=self.s_err[i], sd=1.)
                pm.Normal('b_err_toy', mu=self.b_err[i], sd=1.)
                step = pm.Metropolis() # Metropolis-Hasting algorithm
                trace = pm.sample(N_toys, step,cores=cores, chains=chains)
                pseudo_experiments.append(trace)
        return pseudo_experiments
        
    #     obs_toys=[]
    #     s_toys=[]
    #     b_toys=[]
    #     b_err_toys=[]
    #     for i in range(self.nbins):
	   #       obs_toys.append(list(pseudo_experiments[i]['obs_toy'])      )
	   #       s_toys.append( list(pseudo_experiments[i]['s_toy'])         )
	   #       b_toys.append( list(pseudo_experiments[i]['b_toy'])         )
	   #       b_err_toys.append( list(pseudo_experiments[i]['b_err_toy']) )
    #     return obs_toys,s_toys,b_toys,b_err_toys

    # def toy_samples(self,sample,N_toys=10000,cores=4,chains=2):
    #     res = self.toys(N_toys=N_toys,cores=cores,chains=chains)
    #     if sample=='obs':
    #         return res[0][0]
    #     elif sample=='signal':
    #         return res[1][0]
    #     elif sample=='backgr':
    #         return res[2][0]
    #     elif sample=='b_error':
    #         return res[3][0]
    

class stat__model:

	def __init__(self,mu,data):
		self.mu = mu
		self.data = data

	def param_initial_val(self, nuissance_profiled=False):
		theta_0=[]
		data = self.data
		if nuissance_profiled==False:
			mu_0=[]
			for i in range(data.nbins):
				if data.signal[i]>0:
					rand=random.randint(1,int(data.signal[i]))
				else:
					rand=0
				mu_0.append((data.obs[i]-data.backgr[i]+rand) / data.signal[i])
				theta_0.append((data.obs[i]-data.backgr[i]-data.signal[i]+rand) / data.b_err[i])
			res=[(max(mu_0)-min(mu_0))/2] + theta_0
		else:
			for i in range(data.nbins):
				if data.signal[i]>0:
					rand=random.randint(1,int(data.signal[i]))
				else:
					rand=0
				theta_0.append((data.obs[i]-data.backgr[i]-self.mu*data.signal[i]+rand) / data.b_err[i])
			res=theta_0
		return res

	def Neg_Log_Likelihood(self,params):
		nll=0.
		data = self.data
		mu_0=params[0]
		theta_0=params[1:]
		for i in range(data.nbins):
			rate = mu_0*data.signal[i] + data.backgr[i] + theta_0[i]*data.b_err[i]
			nll += poisson.logpmf(data.obs[i], rate) - 0.5*(theta_0[i])**2 
		return (-1) * nll

	def Neg_Log_Likelihood_cond(self,params):
		nll=0.
		data = self.data
		theta_0=params
		for i in range(data.nbins):
			rate = self.mu*data.signal[i] + data.backgr[i] + theta_0[i]*data.b_err[i]
			nll += poisson.logpmf(data.obs[i], rate) - 0.5*(theta_0[i])**2  
		return (-1) * nll

	def Neg_Log_Likelihood_cond_null(self,params):
		nll=0.
		data = self.data
		theta_0=params
		for i in range(data.nbins):
			rate = data.signal[i] + data.backgr[i] + theta_0[i]*data.b_err[i]
			nll += poisson.logpmf(data.obs[i], rate) - 0.5*(theta_0[i])**2  
		return (-1) * nll


	def minima(self, profiled):
		param = self.param_initial_val(nuissance_profiled=profiled)
		if profiled==False:
			minimum = minimize(self.Neg_Log_Likelihood,param)
		elif profiled==True:
			minimum = minimize(self.Neg_Log_Likelihood_cond,param)
		elif profiled=='null':
			minimum = minimize(self.Neg_Log_Likelihood_cond_null,param)
		return [x for x in minimum.x]

	# def mu_hat(self):
	# 	res=self.minima(profiled=False)
	# 	return res[0]
	# def theta_hat(self):
	# 	res=self.minima(profiled=False)	
	# 	return res[1:]
	# def theta_hat_hat(self):
	# 	res=self.minima(profiled=True)
	# 	return res
	# def theta_hat_hat_null(self):
	# 	res=self.minima(profiled='null')
	# 	return res

	# def test_statistic(self, test_type):  # 'q_0_~', 'q_mu_~', 't_mu_~'

	# 	param_num=[self.mu] + self.theta_hat_hat()
		
	# 	if self.mu_hat()>=0.0:
	# 		param_denom=[self.mu_hat()] + self.theta_hat()
	# 	else:
	# 		param_denom=[0.0] + self.theta_hat_hat()

	# 	log_lambda_tilde = - self.Neg_Log_Likelihood(param_num) + self.Neg_Log_Likelihood(param_denom) 

	# 	if test_type=='q_mu':
	# 		if self.mu_hat() <= self.mu:
	# 			res=-2*log_lambda_tilde 
	# 		else:
	# 			res=0.

	# 	if test_type=='t_mu':
	# 		res=-2*log_lambda_tilde 

		return res

def elapsed_time(t_0):
	t=timer()-t_0
	print('\n')
	if t < 60.: 
		res='time elapsed: {} sec'.format(t)
	elif (t > 60. and t < 3600.0): 
		res='time elapsed: {} min'.format(t/60.)
	elif  t >= 3600.0: 
		res='time elapsed: {} hours'.format(t/3600.)
return print(res)

def test_stat_file(file_name, Mu):
	file = open(file_name+'.dat',"w")
	file.write("<mu>\t" )
	for mu in Mu:
		file.write("%s\t" % mu)
	file.write("\n")
	return file




class toy_experiments:

	print('\n')
	print('Performing toy experiments zzz....')
	results.write("<mu>\n" )
	for mu in Mu:
		results.write("%s\t" % mu)
	results.write("\n")
	results.write("</mu>\n")

	for j in range(num_toys):
		stat_test=[]
		for mu in Mu:
			obs_new=[]
			s_new=[]
			b_new=[]
			b_err_new=[]
			for i in range(data.nbins):
				obs_new.append( pseudo_experiments[i]['obs_toy'][j])
				s_new.append( pseudo_experiments[i]['s_toy'][j])
				b_new.append( pseudo_experiments[i]['b_toy'][j])
				b_err_new.append( pseudo_experiments[i]['b_err_toy'][j])
			data=data__model(nobs=obs_new,
							 s=s_new,
							 b=b_new,
							 s_err=None,
							 b_err=b_err_new)
			model = stat__model(mu,data)
			hat_estimators = model.minima(profiled=False)
			theta_hat_hat = model.minima(profiled=True)
			theta_hat_hat_null = model.minima(profiled='null')
			stat_test.append(q_tilde__statistic(mu,hat_estimators,theta_hat_hat,theta_hat_hat_null,data))
		for st in stat_test:
			results.write("%s\t" % st)
		results.write("\n")

