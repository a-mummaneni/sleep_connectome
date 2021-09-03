hist_generation.py
task = 'PSQI_AmtSleep'

def r_stats(behav_obs_pred, tail="glm"):
    x = behav_obs_pred.filter(regex=("obs")).astype(float)
    x = x['{} observed'.format(task)]
    y = behav_obs_pred.filter(regex=(tail)).astype(float)
    y = y['{} predicted (glm)'.format(task)]
    hm = head_motion['Motion']
    data = {'x': x, 'y': y, 'hm': hm}
    df = pd.DataFrame(data)
    r = pg.partial_corr(data=df, x='x', y='y', covar='hm', method='pearson')
    r_val = r['r'].item()
    p_val = r['p-val'].item()
    print(p_val)
    
    return r_val
    

#actual scores
hist = []

i=1
while i <= 1000:
	cpm_kwargs = {'r_thresh': 0.1, 'corr_type': 'pearson'} 
	behav_obs_pred, all_masks = cpm_wrapper(all_fc_data, all_behav_data, behav=task, **cpm_kwargs)
	r = r_stats(behav_obs_pred)
	hist.append(r)
	i += 1
	print(i)



#randomized shuffled scores
null_hist = []

subject = all_behav_data.index


i=1
while i <= 1000:
	np.seterr(divide='ignore', invalid='ignore')
	b = all_behav_data.sample(frac=1)
	b['Subject'] = subject
	b.set_index('Subject', inplace=True)
	cpm_kwargs = {'r_thresh': 0.01, 'corr_type': 'pearson'}
	behav_obs_pred, all_masks = cpm_wrapper(all_fc_data, b, task, **cpm_kwargs)
	r = r_stats(behav_obs_pred)
	null_hist.append(r)
	i += 1
	print(i)




#plot
plt.hist(hist, alpha=0.5, label=task, color='red')
plt.hist(null_hist, alpha=0.5, label='Shuffled {}'.format(task), color='blue')
plt.xlabel('r')
plt.ylabel('Frequency')
plt.grid(True)

#plt.annotate('p = .03', xy = (.05, .9), xycoords = 'axes fraction')
plt.legend(loc='upper right')
plt.show()



#get p_value
mean = statistics.mean(hist)
greater = [i for i in hist2 if i >= mean]
p_val = len(greater) / 301
print(p_val)









