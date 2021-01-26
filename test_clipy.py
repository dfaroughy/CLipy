import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from clipy import data__model, stat__model
from timeit import default_timer as timer
from scipy.stats import poisson, norm
plt.style.use('ggplot')


#=======================================================================
#                Toy analysis from CMS NOTE-2017/001
#=======================================================================
data = data__model(
				nobs  = [1964,877,354,182,82,36,15,11],
				s     = [47.0,29.4,21.1,14.3,9.4,7.1,4.7,4.3],
				b     = [2006.4,836.4,350.0,147.1,62.0,26.2,11.1,4.7],
				s_err = None,
				b_err = [129.6,24.6,38.1,29.5,18.8,11.0,6.2,3.3] # no correlations
				)

# signal strength range:
Mu = [0.,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4]
# number of pseudo-experiments:
num_toys=10000
# test statistic:
test='q_mu'
#=======================================================================

def Log_Likelihood(mu,theta,dat):
	ll=0.
	for i in range(dat.nbins):
		rate = mu*dat.signal[i] + dat.backgr[i] + theta[i]*dat.b_err[i]
		ll += poisson.logpmf(dat.obs[i], rate) - 0.5*(theta[i])**2 
	return ll

def q_tilde__statistic(mu,hat_param,theta_hat_hat,theta_hat_hat_null,dat):
	num=Log_Likelihood(mu,theta_hat_hat,dat)
	if hat_param[0] < 0.0:
		log_lambda_tilde = num - Log_Likelihood(0.,theta_hat_hat_null,dat) 
	else:
		if hat_param[0] <= mu:
			log_lambda_tilde = num - Log_Likelihood(hat_param[0],hat_param[1:],dat)
		else:
			log_lambda_tilde = -0.
	return (-2)*log_lambda_tilde 

def t_tilde__statistic(mu,hat_param,theta_hat_hat,theta_hat_hat_null,dat):
	num=Log_Likelihood(mu,theta_hat_hat,dat)
	if hat_param[0] < 0.0:
		log_lambda_tilde = num - Log_Likelihood(0.,theta_hat_hat_null,dat) 
	else:
		log_lambda_tilde = num - Log_Likelihood(hat_param[0],hat_param[1:],dat)
	return (-2)*log_lambda_tilde 

# stat_test=[]
# for mu in Mu:
# 	model = stat__model(mu,data)
# 	hat_estimators=model.minima(profiled=False)
# 	theta_hat_hat=model.minima(profiled=True)
# 	theta_hat_hat_null=model.minima(profiled='null')
# 	stat_test.append(q_tilde__statistic(mu,hat_estimators,theta_hat_hat,theta_hat_hat_null,data))
# 	# model = stat__model(mu,data)
# 	# stat_test.append(model.test_statistic(test_type=test))

# Mu_new = np.linspace(0., max(Mu), num=60, endpoint=True)
# stat_test_func = interp1d(Mu, stat_test, kind='cubic')


# Toy experiments .......................

pseudo_experiments = data.toys(N_toys=num_toys,cores=4,chains=1)
print('\n')
print('Performing toy experiments zzz....')


results=test_stat_file(file_name, Mu)
start = timer()

for j in range(num_toys):
	stat_test=[]
	for mu in Mu:
		obs_new=[];s_new=[];b_new=[];b_err_new=[]
		for i in range(data.nbins):
			obs_new.append( pseudo_experiments[i]['obs_toy'][j])
			s_new.append( pseudo_experiments[i]['s_toy'][j])
			b_new.append( pseudo_experiments[i]['b_toy'][j])
			b_err_new.append( pseudo_experiments[i]['b_err_toy'][j])
		
		data_obs=data__model(nobs=obs_new,
						 s=s_new,
						 b=b_new,
						 s_err=None,
						 b_err=b_err_new) 

		model = stat__model(mu,data)

		hat_estimators = model.minima(profiled=False)
		theta_hat_hat = model.minima(profiled=True)
		theta_hat_hat_null = model.minima(profiled='null')

		stat_test_s_plus_b.append(q_tilde__statistic(mu,hat_estimators,theta_hat_hat,theta_hat_hat_null,data))
		stat_test_b.append(q_tilde__statistic(mu,hat_estimators,theta_hat_hat,theta_hat_hat_null,data))

	for st in stat_test:
		results.write("%s\t" % st)
	results.write("\n")
elapsed_time(t_0=start)

Mus=[]
with open('./q_mu.dat', 'r') as data:
	next(data)
	for line in data:
		Mus.append( [float(l) for l in str.split(line)] )
Mus=list(map(list, zip(*Mus)))




# # Plot............................

# q_mu:

fig = plt.figure(figsize=(5,5))
ax = plt.subplot(111)
plt.scatter(x=Mu, y=stat_test, s=30, c='orange', alpha=1)
plt.plot(Mu, stat_test_func(Mu), '-', color='orange')
plt.hlines(1, 0.0, max(Mu), colors='r', ls='--',lw=0.75)
plt.xlabel(r'Signal strength $\mu$')
plt.ylabel(r'test statistic $(\tilde t_\mu, \tilde q_\mu)$')
ax.set_xlim([0., max(Mu)])
ax.set_ylim([min(stat_test)-0.2,max(stat_test)])
plt.tight_layout()
plt.savefig('./plots/'+test+'_statistic.pdf')

# test statistic plots from toys:

bins=range(40)
fig = plt.figure(figsize=(7,7))
ax = plt.subplot(111)
for m in Mus:
	plt.hist(m, bins, histtype='step', normed=True, lw=1.5);
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([0., 40])
plt.tight_layout()
plt.savefig('./plots/test_stat.pdf')

# # toy experiment plots:

# Nobs_toys=[]
# S_toys=[]
# B_toys=[]
# DB_toys=[]

# for I in range(data.nbins):
# 	Nobs_toys.append( list(pseudo_experiments[I]['obs_toy']) )  
# 	S_toys.append( list(pseudo_experiments[I]['s_toy']) )
# 	B_toys.append( list(pseudo_experiments[I]['b_toy']) )
# 	DB_toys.append( list(pseudo_experiments[I]['b_err_toy']) )

# for I in range(data.nbins):
# 	Nobs_toys.append( list(pseudo_experiments[I]['obs_toy']) )  
# 	S_toys.append( list(pseudo_experiments[I]['s_toy']) )
# 	B_toys.append( list(pseudo_experiments[I]['b_toy']) )
# 	DB_toys.append( list(pseudo_experiments[I]['b_err_toy']) )

# fig2 = plt.figure(figsize=(7,14))

# ax1 = plt.subplot(411)
# plt.hist(Nobs_toys[0], 30, histtype='step', normed=True);
# plt.hist(Nobs_toys[1], 30, histtype='step', normed=True);
# plt.hist(Nobs_toys[2], 30, histtype='step', normed=True);
# plt.hist(Nobs_toys[3], 30, histtype='step', normed=True);
# plt.hist(Nobs_toys[4], 30, histtype='step', normed=True);
# plt.hist(Nobs_toys[5], 30, histtype='step', normed=True);
# plt.hist(Nobs_toys[6], 30, histtype='step', normed=True);
# plt.hist(Nobs_toys[7], 30, histtype='step', normed=True);
# plt.title('Nobs toys');

# ax2 = plt.subplot(412)
# plt.hist(B_toys[0], 30, histtype='step', normed=True);
# plt.hist(B_toys[1], 30, histtype='step', normed=True);
# plt.hist(B_toys[2], 30, histtype='step', normed=True);
# plt.hist(B_toys[3], 30, histtype='step', normed=True);
# plt.hist(B_toys[4], 30, histtype='step', normed=True);
# plt.hist(B_toys[5], 30, histtype='step', normed=True);
# plt.hist(B_toys[6], 30, histtype='step', normed=True);
# plt.hist(B_toys[7], 30, histtype='step', normed=True);
# plt.title('Background toys');

# ax3 = plt.subplot(413)
# plt.hist(S_toys[0], 10, histtype='step', normed=True);
# plt.hist(S_toys[1], 10, histtype='step', normed=True);
# plt.hist(S_toys[2], 10, histtype='step', normed=True);
# plt.hist(S_toys[3], 10, histtype='step', normed=True);
# plt.hist(S_toys[4], 10, histtype='step', normed=True);
# plt.hist(S_toys[5], 10, histtype='step', normed=True);
# plt.hist(S_toys[6], 10, histtype='step', normed=True);
# plt.hist(S_toys[7], 10, histtype='step', normed=True);
# plt.title('Signal toys');

# ax4 = plt.subplot(414)
# plt.hist(DB_toys[0], 30, histtype='step', normed=True);
# plt.hist(DB_toys[1], 30, histtype='step', normed=True);
# plt.hist(DB_toys[2], 30, histtype='step', normed=True);
# plt.hist(DB_toys[3], 30, histtype='step', normed=True);
# plt.hist(DB_toys[4], 30, histtype='step', normed=True);
# plt.hist(DB_toys[5], 30, histtype='step', normed=True);
# plt.hist(DB_toys[6], 30, histtype='step', normed=True);
# plt.hist(DB_toys[7], 30, histtype='step', normed=True);
# plt.title('background uncertainty toys');

# plt.tight_layout()
# plt.savefig('./plots/toy_experiments.pdf')



