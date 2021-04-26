import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var
from torch.autograd import Variable


class SentenceVaeStyleOrtho(nn.Module):
	def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
				sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1, bidirectional=False):

		super().__init__()
		self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

		
		self.content_bow_dim = 7526
		self.max_sequence_length = max_sequence_length

		self.sos_idx = sos_idx
		self.eos_idx = eos_idx
		self.pad_idx = pad_idx
		self.unk_idx = unk_idx

		self.latent_size = latent_size

		self.rnn_type = rnn_type
		# self.bidirectional = bidirectional
		self.bidirectional = False
		self.num_layers = num_layers
		self.hidden_size = hidden_size 
		self.output_size = 2

		self.embedding = nn.Embedding(vocab_size, embedding_size)
		self.word_dropout_rate = word_dropout
		self.embedding_dropout = nn.Dropout(p=embedding_dropout)

		if rnn_type == 'rnn':
			rnn = nn.RNN
		elif rnn_type == 'gru':
			rnn = nn.GRU
		else:
			raise ValueError()

		self.lstm = nn.LSTM(embedding_size, hidden_size)

		self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
		self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)

		self.hidden_factor = (2 if bidirectional else 1) * num_layers
		
		# self.label = nn.Linear(hidden_size, self.output_size)


		#####################################################################################

		# hidden to style space
		self.hidden2stylemean = nn.Linear(hidden_size * self.hidden_factor, int(latent_size/4))
		self.hidden2stylelogv = nn.Linear(hidden_size * self.hidden_factor, int(latent_size/4))

		# hidden to content space
		self.hidden2contentmean = nn.Linear(hidden_size * self.hidden_factor, int(3*latent_size/4))
		self.hidden2contentlogv = nn.Linear(hidden_size * self.hidden_factor, int(3*latent_size/4))

		# classifiers
		self.content_classifier = nn.Linear(int(3*latent_size/4), self.content_bow_dim)
		self.style_classifier_1 = nn.Linear(int(latent_size/4), self.output_size) # for correlating style space to sentiment
		# self.style_classifier_2 = nn.Linear(10, 2) # for correlating style space to sentiment

		self.style_sigmoid = nn.ReLU()

		# dsicrimimnator/adversaries

		# latent to initial hs for decoder
		self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)

		# final hidden to output vocab
		self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

		#extra parameters for style disentg.
		
		self.label_smoothing = 0.1
		self.num_style = 2
		self.dropout_rate = 0.5

		self.dropout = nn.Dropout(self.dropout_rate)

	def forward(self, input_sequence, length, labels, content_bow):

		batch_size = input_sequence.size(0) #get batch size
		sorted_lengths, sorted_idx = torch.sort(length, descending=True) #sort input sequences into inc order
		# input_sequence = input_sequence[sorted_idx] #get sorted sentences

		input_embedding = self.embedding(input_sequence) # convert to embeddings
		input_embedding = input_embedding.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)

		h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
		c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())

		#pad inputs to uniform length
		# packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True) #(B, L, E)

		# run encoder
		output, (hidden, final_cell_state) = self.lstm(input_embedding, (h_0, c_0))
		hidden = hidden[-1] # take the last hidden state of lstm

		# _, hidden = self.encoder_rnn(packed_input) # hidden -> (B, H)

		# if the RNN has multiple layers, flatten all the hiddens states 
		if self.bidirectional or self.num_layers > 1:
		    hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor) # flatten hidden state
		else:
			hidden = hidden.squeeze()

		
		#encoder RNN done, hidden now contains the final hidden states to be mapped into prob dist.

		# REPARAMETERIZATION of style and content

		#style component

		style_mean = self.hidden2stylemean(hidden) #calc latent mean 
		style_logv = self.hidden2stylelogv(hidden) #calc latent variance
		style_std = torch.exp(0.5 * style_logv) #find sd

		style_z = to_var(torch.randn([batch_size, int(self.latent_size/4)])) #get a random vector
		style_z = style_z * style_std + style_mean #compute datapoint

		#content component

		content_mean = self.hidden2contentmean(hidden) #calc latent mean 
		content_logv = self.hidden2contentlogv(hidden) #calc latent variance
		content_std = torch.exp(0.5 * content_logv) #find sd

		content_z = to_var(torch.randn([batch_size, int(3*self.latent_size/4)])) #get a random vector
		content_z = content_z * content_std + content_mean #compute datapoint


		# #concat style and concat

		final_mean = torch.cat((style_mean, content_mean), dim=1)
		final_logv = torch.cat((style_logv, content_logv), dim=1)
		final_z = torch.cat((style_z, content_z), dim=1)

		# #style and content classifiers

		# run style classifier on hidden space (proposal 1)
		style_preds = self.style_classifier_1(style_z) # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)

		
		# style_mul_loss, style_preds = self.get_style_mul_loss(style_z, labels, batch_size)
		content_mul_loss = self.get_content_mul_loss(content_z, content_bow)

		# # DECODER
		hidden = self.latent2hidden(final_z)

		if self.bidirectional or self.num_layers > 1:
		    # unflatten hidden state
		    hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
		else:
		    hidden = hidden.unsqueeze(0)

		# decoder input
		if self.word_dropout_rate > 0:
			
		    # randomly replace decoder input with <unk>
		    prob = torch.rand(input_sequence.size())
			
		    if torch.cuda.is_available():
		        prob=prob.cuda()
		    prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
		    decoder_input_sequence = input_sequence.clone()
		    decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
		    input_embedding = self.embedding(decoder_input_sequence)

		# input_embedding = self.embedding_dropout(input_embedding)
		input_embedding = input_embedding.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)
		packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

		# decoder forward pass
				
		outputs, _ = self.decoder_rnn(packed_input, hidden)

		# process outputs
		padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
		padded_outputs = padded_outputs.contiguous()
		_,reversed_idx = torch.sort(sorted_idx)
		padded_outputs = padded_outputs[reversed_idx]
		b,s,_ = padded_outputs.size()

		# project outputs to vocab
		logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
		logp = logp.view(b, s, self.embedding.num_embeddings)

		return logp, final_mean, final_logv, final_z, content_mul_loss, style_preds

		
	def get_content_mul_loss(self, content_emb, content_bow):
		"""
		This loss quantifies the amount of content information preserved
		in the content space
		Returns:
		cross entropy loss of the content classifier
		"""
		# predictions
		preds = nn.Softmax(dim=1)(self.content_classifier(self.dropout(content_emb)))
		
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
			output, hidden = self.decoder_rnn(input_embedding, hidden)
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
