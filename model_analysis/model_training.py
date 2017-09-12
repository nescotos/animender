from sae import SAE
import torch

training_set = torch.load('./training_set.pkl')
test_set = torch.load('./test_set.pkl')
sae = SAE(3787, encoder_input=40, decoder_input=40)
sae.add_hiden_layer(20)
sae.add_dropout(0.2)
sae.add_hiden_layer(40)
sae.compile(optimizer='adam')
sae.fit(training_set, 5)
sae.perform(training_set, test_set)
torch.save(sae, 'model.pkl')
