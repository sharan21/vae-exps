import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var
from torch.autograd import Variable


class SentenceVaeStyleOrtho(nn.Module):
	def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
				sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1, bidirectional=False, ortho=False, 
				attention=False, hspace_classifier=False, diversity=False):

		super().__init__()

		
		self.ortho = ortho
		self.attention = attention
		self.hspace_classifier = hspace_classifier
		self.diversity = diversity
		self.diversity_weight = 10 # if diversity loss is enabled, this is the scaling factor, more diversity, more orthogonalisation

		self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

		self.content_bow_dim = 7526
		self.max_sequence_length = max_sequence_length

		self.sos_idx = sos_idx
		self.eos_idx = eos_idx
		self.pad_idx = pad_idx
		self.unk_idx = unk_idx

		self.latent_size = latent_size
		self.style_content_split = 0.5 #if this is not 0.5, we cannot take dot product of style_z and content_z!

		self.rnn_type = rnn_type
		self.bidirectional = bidirectional # need to be implemented
		# self.bidirectional = False
		self.num_layers = num_layers
		self.hidden_size = hidden_size 
		self.output_size = 2

		self.embedding = nn.Embedding(vocab_size, embedding_size)
		self.word_dropout_rate = word_dropout
		self.embedding_dropout = nn.Dropout(p=embedding_dropout)
		
		if(self.style_content_split != 0.5 and (self.ortho or self.diversity)):
			print("does not suppose style:content split as non 0.5 yet!")
			exit()
		
		if(self.ortho is True and self.diversity is True): 
			print("you have enabled both GS orthoganlisation and Diversity loss, pick one!")
			exit()

		if rnn_type == 'rnn':
			rnn = nn.RNN
		elif rnn_type == 'gru':
			rnn = nn.GRU
		else:
			raise ValueError()

		self.encoder = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
		self.decoder = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
		self.hidden_factor = (2 if bidirectional else 1) * num_layers

		######## add hspace classifier #######
		if(self.hspace_classifier):
			self.hspace_classifier_layer = nn.Linear(self.hidden_size * self.hidden_factor, self.output_size)	

		######## hidden to style space ########
		self.hidden2stylemean = nn.Linear(hidden_size * self.hidden_factor, int(latent_size/2))
		self.hidden2stylelogv = nn.Linear(hidden_size * self.hidden_factor, int(latent_size/2))

		######### hidden to content space#######
		self.hidden2contentmean = nn.Linear(hidden_size * self.hidden_factor, int(latent_size/2))
		self.hidden2contentlogv = nn.Linear(hidden_size * self.hidden_factor, int(latent_size/2))

		########## classifiers ############
		self.content_classifier = nn.Linear(int(latent_size/2), self.content_bow_dim)
		self.style_classifier = nn.Linear(int(latent_size/2), self.output_size) # for correlating style space to sentiment
		
		############ adversaries ###########
		# need to add these

		######### latent to initial hs for decoder ########
		self.latent2hidden = nn.Linear(latent_size, self.hidden_size * self.hidden_factor)

		###### final hidden to output vocab #########
		self.outputs2vocab = nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), vocab_size)

		###### extra misc parameters ########
		
		self.label_smoothing = 0.1
		self.num_style = 2
		self.dropout_rate = 0.5
		
		self.dropout = nn.Dropout(self.dropout_rate)
  
	def self_attention(self, lstm_output, final_state):

		# lstm_output : L*B*H
		# final_state : L*B*1
		
		# reshaping to satisfy torch.bmm
		lstm_output_2 = lstm_output.permute(1,0,2) #B*L*H
		final_state_2 = final_state.permute(1,2,0) #B*1*L
		
		#get attention scores
		attn_weights = torch.bmm(lstm_output_2, final_state_2) #B*L, dot product attention
		soft_attn_weights = F.softmax(attn_weights, 1) #B*L
		
		# weighted sum to get final attention vector
		lstm_output_2 = lstm_output.permute(1,0,2) #B*L*H, needed for mat mul in next step
		new_hidden_state = lstm_output_2 * soft_attn_weights #B*L*H * B*L = #B*L*H
		new_hidden_state = torch.sum(new_hidden_state, axis=1) #B*H
		
		return new_hidden_state


	def forward(self, input_sequence, length, labels, content_bow):

		batch_size = input_sequence.size(0) #get batch size
		input_embedding = self.embedding(input_sequence) # convert to embeddings
		
		################### encoder ##################
		_, hidden = self.encoder(input_embedding) # hidden -> (B, H)
	
		###### if the RNN has multiple layers, flatten all the hiddens states 
		if self.bidirectional or self.num_layers > 1:
		    hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor) # flatten hidden state
		else:
			hidden = hidden.squeeze()

		####### self attention
		if(self.attention):
			if(self.style_content_split != 0.5):
				print("Cannot orthogonalise style and content embeddings of different dims!")
				exit()
			hidden = self.self_attention(output, hidden)
		# else:
			# hidden = hidden[-1] # take the last hidden state of lstm

		####### if the RNN has multiple layers, flatten all the hiddens states 
		if self.bidirectional or self.num_layers > 1:
			hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor) # flatten hidden state
		else:
			hidden = hidden.squeeze()

		###################### proposal 1: hspace classifier ##################
		if(self.hspace_classifier):
			hspace_preds = self.hspace_classifier_layer(hidden)
		else:
			hspace_preds = None

		############## REPARAMETERIZATION of style and content###############

		############style component

		style_mean = self.hidden2stylemean(hidden) #calc latent mean 
		style_logv = self.hidden2stylelogv(hidden) #calc latent variance
		style_std = torch.exp(0.5 * style_logv) #find sd

		style_z = to_var(torch.randn([batch_size, int(self.latent_size/2)])) #get a random vector
		style_z = style_z * torch.exp(style_logv) + style_mean #copmpute datapoint

		############content component

		content_mean = self.hidden2contentmean(hidden) #calc latent mean 
		content_logv = self.hidden2contentlogv(hidden) #calc latent variance
		content_std = torch.exp(0.5 * content_logv) #find sd

		content_z = to_var(torch.randn([batch_size, int(self.latent_size/2)])) #get a random vector
		content_z = content_z * torch.exp(content_logv) + content_mean #compute datapoint

		####################### orthogonalise content_z w.r.t style_z using Gram schmitt
		
		if(self.ortho): #from https://arxiv.org/abs/2004.14243
		
			u_v = style_z*content_z
			u_v = torch.sum(u_v, axis=-1).unsqueeze(-1)
			u_u = style_z*style_z
			u_u = torch.sum(u_u, axis=-1).unsqueeze(-1)
			
			content_z = content_z - u_v/u_u * style_z #orthogonalised content


		####################### orthogonalise content_z w.r.t style_z using Gram schmitt
		
		if(self.diversity): #from https://arxiv.org/abs/1704.08300 and https://arxiv.org/abs/2004.14243
			diversity_loss = style_z*content_z # B*latent_size
			diversity_loss = torch.sum(diversity_loss, axis=-1) #B
			diversity_loss = torch.mean(diversity_loss, axis=0) #1
		else:
			diversity_loss = 0

		#############concat style and concat
		
		final_mean = torch.cat((style_mean, content_mean), dim=1)
		final_logv = torch.cat((style_logv, content_logv), dim=1)
		final_z = torch.cat((style_z, content_z), dim=1)

		##########3#####style and content classifiers###################
		
		style_preds = self.style_classifier(style_z)
		content_preds = self.content_classifier(content_z)

		######################## DECODER
		hidden = self.latent2hidden(final_z)

		if self.bidirectional or self.num_layers > 1:
		    # unflatten hidden state
		    hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
		else:
		    hidden = hidden.unsqueeze(0)

		############# decoder input
		if self.word_dropout_rate > 0:
			
		    # randomly replace decoder input with <unk>
		    prob = torch.rand(input_sequence.size())
			
		    if torch.cuda.is_available():
		        prob=prob.cuda()

		    prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
		    decoder_input_sequence = input_sequence.clone()
		    decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
		    input_embedding = self.embedding(decoder_input_sequence)

		############### decoder forward pass
		outputs, _ = self.decoder(input_embedding, hidden)

		outputs = outputs.contiguous()
		b,s,_ = outputs.size()

		################ project outputs to vocab
		logp = nn.functional.log_softmax(self.outputs2vocab(outputs.view(-1, outputs.size(2))), dim=-1)
		logp = logp.view(b, s, self.embedding.num_embeddings)

		return logp, final_mean, final_logv, final_z, style_preds, content_preds, hspace_preds, diversity_loss



	def get_content_mul_loss(self, content_emb, content_bow):
		"""
		This loss quantifies the amount of content information preserved
		in the content space
		Returns:
		cross entropy loss of the content classifier
		"""
		# predictions
		preds = nn.Softmax(dim=1)(self.content_classifier(self.dropout(content_emb)))
		# print(content_bow[0])
		
		# label smoothing
		smoothed_content_bow = content_bow * (1-self.label_smoothing) + self.label_smoothing/self.content_bow_dim
		# calculate cross entropy loss
		content_mul_loss = nn.BCELoss()(preds, smoothed_content_bow)

		return content_mul_loss

	
	def inference(self, n=4, z=None):

		if z is None:
			batch_size = n
			z = to_var(torch.randn([batch_size, self.latent_size]))
		else:
			batch_size = z.size(0)

		hidden = self.latent2hidden(z)

		if self.bidirectional or self.num_layers > 1:
			# unflatten hidden state
			hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

		hidden = hidden.unsqueeze(0)

		# required for dynamic stopping of sentence generation
		sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long()  # all idx of batch
		# all idx of batch which are still generating
		sequence_running = torch.arange(0, batch_size, out=self.tensor()).long()
		sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()
		# idx of still generating sequences with respect to current loop
		running_seqs = torch.arange(0, batch_size, out=self.tensor()).long()

		generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

		t = 0
		while t < self.max_sequence_length and len(running_seqs) > 0:

			if t == 0:
				input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())

			input_sequence = input_sequence.unsqueeze(1)
			input_embedding = self.embedding(input_sequence)
			output, hidden = self.decoder(input_embedding, hidden)
			logits = self.outputs2vocab(output)
			input_sequence = self._sample(logits)

			# save next input
			generations = self._save_sample(generations, input_sequence, sequence_running, t)

			# update gloabl running sequence
			sequence_mask[sequence_running] = (input_sequence != self.eos_idx)
			sequence_running = sequence_idx.masked_select(sequence_mask)

			# update local running sequences
			running_mask = (input_sequence != self.eos_idx).data
			running_seqs = running_seqs.masked_select(running_mask)

			# prune input and hidden state according to local update
			if len(running_seqs) > 0:
				input_sequence = input_sequence[running_seqs]
				hidden = hidden[:, running_seqs]

				running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

			t += 1

		return generations, z

	def _sample(self, dist, mode='greedy'):

		if mode == 'greedy':
			_, sample = torch.topk(dist, 1, dim=-1)
		sample = sample.reshape(-1)

		return sample

	def _save_sample(self, save_to, sample, running_seqs, t):
		# select only still running
		running_latest = save_to[running_seqs]
		# update token at position t
		running_latest[:,t] = sample.data
		# save back
		save_to[running_seqs] = running_latest

		return save_to

	def encode_to_lspace(self, input_sequence, length):

		batch_size = input_sequence.size(0) #get batch size
		sorted_lengths, sorted_idx = torch.sort(length, descending=True) #sort input sequences into inc order
		input_sequence = input_sequence[sorted_idx] #get sorted sentences

		input_embedding = self.embedding(input_sequence) # convert to embeddings

		#pad inputs to uniform length
		packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True) #(B, L, E)

		_, hidden = self.encoder_rnn(packed_input) # hidden -> (B, H)

		# if the RNN has multiple layers, flatten all the hiddens states 
		if self.bidirectional or self.num_layers > 1:
			# flatten hidden state
			hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
		else:
			hidden = hidden.squeeze()

		#encoder RNN done, hidden now contains the final hidden states to be mapped into prob dist.

		# REPARAMETERIZATION
		mean = self.hidden2mean(hidden) #calc latent mean 
		logv = self.hidden2logv(hidden) #calc latent variance
		std = torch.exp(0.5 * logv) #find sd

		z = to_var(torch.randn([batch_size, self.latent_size])) #get a random vector
		z = z * std + mean #compute datapoint

		return z

	def conicity_measure(self, styles, contents):
    	# conicity is defined as the mean of ATM/ alignment to mean vector
		# styles, contents: B*latent_size/2
		# conicities: B

		mean_vector = (styles+contents)/2
		style_atm = torch.sum(styles*mean_vector, axis = 1)
		content_atm = torch.sum(contents*mean_vector, axis = 1)
		conicities = (style_atm+content_atm)/2

		return conicities