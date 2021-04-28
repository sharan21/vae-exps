# have done
	1. lit survey in GANs and VAEs **
	2. have code for sandeep et al.
	3. have code for Sentence VAE and PTB dataset for showing interpolations
	4. have code for arae
	6. added style transfer to yelps
	7. replicating "disentagled" paper in pytorch from scratch

# To do
	

## Replicating baseline paper: https://www.aclweb.org/anthology/P19-1041.pdf

		1. add yelp dataset (done)
		2. add snli dataset (doing)
		3. add sentiment classifier (done)
		4. add content classifier (done)
		5. add adversaries (optional, later)
    
## Proposal 1 (Training a classifier on hidden states to boost performance)

		1. add classifier over hidden space (done)
		2. add self attention (done, fix bugs)
		3. add tsne function  (later)
		4. compare performances, ablation study (later)

	

## Proposal 2 (An orthogonalisation based approach to disentanglement)

		1. add gram schmitt ortho style_z and content_z (done)
		2. add diversity loss (done)	
		3. add conicity plot (done, fix bugs)
		4. add demo for arithmetic on orthogonal lspace (later)
		4. add ablation studies (later)

	

## Proposal 3 (Towards a general approach of style transfer for multiple tasks)

		1. add SNLI dataset preprocessing, option 1+2 (done)
		2. add SNLI model and POC of style transfer (done)
		3. add multi task style classifier with yelp + snli
		4. add inter class style transfer

## Misc bugs:
		0. fix NLL loss bug (done)
		1. ensure that bow model is trained for each indv dataset 
		2. diversity loss is -ve, is that okay? 
		3. fix the style and content split ratio (done) 
		4. enable bidirectional encoder (done)
		5. fix bugs in inference.py, make it autoload using params.json (done)
		6. fix sentence repetition problem in yelp
	
	