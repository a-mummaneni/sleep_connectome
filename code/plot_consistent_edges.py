plot_consistent_edges.py



#this code should be run after both hcp and abcd cpms are run

#load in coordination matrix
shen268_coords = pd.read_csv("shen268_coords.csv", index_col="NodeNo")
print(shen268_coords.shape)
shen268_coords.head()





def plot_consistent_edges(all_masks1, all_masks2, tail, thresh = 1., color='gray'):
	edge_frac = (all_masks1[tail].sum(axis=0))/(all_masks1[tail].shape[0])
	edge_frac2 = (all_masks2[tail].sum(axis=0))/(all_masks2[tail].shape[0])
	data = {'Edge Frac 1' : edge_frac, 'Edge Frac 2' : edge_frac2}
	df = pd.DataFrame(data)

	df.loc[df['Edge Frac 1'] < thresh, 'Edge Frac 1'] = 0
	df.loc[df['Edge Frac 2'] < thresh, 'Edge Frac 1'] = 0 #find only edges that meet threshold in both makss

	edge_frac_new = df['Edge Frac 1'].to_numpy()
	print("For the {} tail, {} edges were selected in at least {}% of folds".format(tail, (edge_frac_new>=thresh).sum(), thresh*100))

	edge_frac_square = sp.spatial.distance.squareform(edge_frac_new)
	nodes = edge_frac_square >= thresh

	node_mask = np.amax(edge_frac_square, axis=0) >= thresh # find nodes that have at least one edge that passes the threshold
	node_size = edge_frac_square.sum(axis=0)*node_mask*.75 # size nodes based on how many suprathreshold edges they have
	plot_connectome(adjacency_matrix=edge_frac_square, edge_threshold=thresh,
                    node_color = color,
                    node_coords=shen268_coords, node_size=node_size,
                    display_mode= 'lzry',
                    edge_kwargs={"linewidth": .25, 'color': color})

	return nodes



plot_consistent_edges(all_masks, all_masks2, "pos", thresh = .8, color = 'red')
plot_consistent_edges(all_masks, all_masks2, "neg", thresh = .8, color = 'blue')

