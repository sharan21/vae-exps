This is the repository for our NLP project (CS5803) implemented from scratch. We replicated the base paper (https://www.aclweb.org/anthology/P19-1041.pdf) in pytorch and modified it with our proposals 1,2 and 3. We have added 3 datasets in this study i.e. Penn Tree Bank (for pure generation), Yelp (For Sentiment classification) and SNLI (for Natural language inference). 

# How to run

## For single dataset generation aka without multitask style transfer (ptb/yelp/snli)

Here is an example on how to download, train and infer from the Yelp dataset.

    bash download_yelp.sh
    python3 train_yelp.py
    python3 inference_yelp.py -n 1 -c 'path/to/checkpoint.pytorch' -p 'path/to/model_params.json'




# Overall Progress
	1. lit survey in GANs and VAEs **
	2. have code for sandeep et al.
	3. have code for Sentence VAE and PTB dataset for showing interpolations
	4. have code for arae
	6. added style transfer to yelps
	7. replicating "disentagled" paper in pytorch from scratch
	8. Implemented proposal 1,2,3

# To do
	
All tasks marked "later" are not essential to the current work

## Replicating baseline paper: 

		1. add yelp dataset (done)
		2. add snli dataset (done)
		3. add sentiment classifier (done)
		4. add content classifier (done)
		5. add adversaries (later)
    
## Proposal 1 (Training a classifier on hidden states to boost performance)

		1. add classifier over hidden space (done)
		2. add self attention (done, fix bugs)
		3. add tsne function  (done)
		4. add bleu scores  (done)
		5. compare performances, ablation study 


## Proposal 2 (An orthogonalisation based approach to disentanglement)

		1. add gram schmitt ortho style_z and content_z (done)
		2. add diversity loss (done)	
		3. add conicity plot (done, need to test)
		4. add demo for arithmetic on orthogonal lspace (later)
		5. compare performances, add ablation studies (later)


## Proposal 3 (Towards a general approach of style transfer for multiple tasks)

		1. add SNLI dataset preprocessing, option 1+2 (done)
		2. add SNLI model and POC of style transfer (done)
		3. add multi task style classifier with yelp + snli (done)
		4. add inter class style transfer (done)
		5. add style grafting (done, testing)

## Misc bugs:
		0. fix NLL loss bug (done)
		1. ensure that bow model is trained for each indv dataset (later)
		2. diversity loss is -ve, is that okay? (done)
		3. fix the style and content split ratio (done) 
		4. enable bidirectional encoder (done)
		5. fix bugs in inference.py, make it autoload using params.json (done)
		6. fix sentence repetition problem in yelp (later)
		7. fix dataloader problem for snli_yelp (done)