# Overall progress
	1. lit survey in GANs and VAEs **
	2. have code for sandeep et al.
	3. have code for Sentence VAE and PTB dataset for showing interpolations
	4. have code for arae
	5. replicating "disenntagled" in pytorch from scratch (ongoing)
	6. adding proposal 1,2 and 3 (ongoing)

# To do
	
	**Replicating baseline paper:** https://www.aclweb.org/anthology/P19-1041.pdf
		0. add yelp dataset (done)
		1. add sentiment classifier (done)
		3. add content classifier (done)
		4. add adversaries (optional, as naveen to do)

	**Proposal 1 (Training a classifier on hidden states to boost performance)**
		1. add classifier over hidden space
		2. add self attention (done)
		3. add tsne function
		4. compare performances, ablation study (sumitra)

	**Proposal 2 (An orthogonalisation based approach to disentanglement)**
		1. orthoganlise style_z and content_z
		2. add diversity loss
		3. add conicity plot
		4. add demo for arithmetic on orthogonal lspace

	**Proposal 3 (Towards a general approach of style transfer for multiple tasks)**
		1. add SNLI dataset preprocessing, option 1+2 (sumitra)
		2. add multi task style classifier with yelp + snli
		3. add inter class style transfer