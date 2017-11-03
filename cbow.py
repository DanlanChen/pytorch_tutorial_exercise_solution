# from the tutorial:http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html#sphx-glr-beginner-nlp-word-embeddings-tutorial-py


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()
embedding_dim = 10
# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)
# print vocab_size
word_to_ix = {word: i for i, word in enumerate(vocab)}
# print word_to_ix
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
# print(data[:5])


class CBOW(nn.Module):

    def __init__(self,vocab_size, embedding_dim,context_size):
        super(CBOW,self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim,vocab_size)
        self.linear2 = nn.Linear(embedding_dim,128)
        self.linear3 = nn.Linear(128,vocab_size)
    def forward(self,inputs):
        # print inputs.size(),'inputs'
        embeds = self.embeddings(inputs)
        # print embeds.size()
        out = embeds.sum(dim = 0)
        # print out.size()
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        log_probs = 1.0 * F.log_softmax(out).view(1,-1)
        return log_probs


# create your model and train.  here are some functions to help you make
# the data ready for use by your module


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

def make_target_vector(w,word_to_ix):
    idx = word_to_ix[w]
    # print idx
    tensor = torch.LongTensor([idx])
    # print tensor
    return autograd.Variable(tensor)
# print data[0][1]
# print data[0][0],make_context_vector(data[0][0], word_to_ix)  # example
# print make_target_vector(data[0][1],word_to_ix)
# print len(data)
losses = []
loss_function = nn.NLLLoss()
model = CBOW(vocab_size,embedding_dim,CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)
for epoch in range(100):
    total_loss = torch.Tensor([0])
    for context, target in data:
        context_var = make_context_vector(context, word_to_ix)
        model.zero_grad()
        log_probs = model(context_var)

        target_var = make_target_vector(target,word_to_ix)
        # print log_probs
        # print log_probs.size(),'log_probs'
        # print target_var.size(),'target_var'
        loss = loss_function(log_probs,target_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    losses.append(total_loss)
print losses
