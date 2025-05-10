import torch
import torch.nn as nn
from torch.nn import functional as F 

# !curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embed = 40
n_layer = 2
n_head = 6
dropout = 0.2

torch.manual_seed(1337) 


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read() 



chars = sorted(list(set(text)))
vocab_size = len(chars)


def create_encode_and_decode():
    stoi = { ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return encode, decode

encode, decode = create_encode_and_decode()


def create_data():
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    block_size = 8
    train_data[:block_size+1]

    x = train_data[:block_size]
    y = train_data[1:block_size+1]
    for t in range(block_size):
        context = x[:t+1]
        target = y[t]
    
    return data, train_data, val_data

data, train_data, val_data = create_data()


def get_batch(split):
    #generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x,y

xb, yb = get_batch('train')

@torch.no_grad() # wont call .backward on, more efficient as it now does not store values for backward propogation
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Example function underlining context
# for b in range(batch_size): # batch dimension
#                for t in range(block_size): # time dimension
#                        context = xb[b, :t+1]
#                        target = yb[b,t]
#                        print(f"When input is {context.tolist()} the target: {target}")



class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2,-1) * C**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        v = self.value(x)
        out = weights @ v
        return out
    
class MultiHeadAttention(nn.Module):
    """ mutliple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

#A bigram language model is a statistical language model 
# used in natural language processing (NLP) to predict the likelihood 
# of a word in a sequence based on the preceding word

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """" Transformer block: communication followed by computation"""

    def __init__(self, n_embed, n_head):
        # n_embed: embedding dimension, n_head: the number of heads we'd like 
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # A token embedding table is a lookup dictionary that converts words or word pieces (tokens) into numerical vectors 
        # that neural networks can understand. It maps each token in a model's vocabulary to a unique vector of numbers that 
        # captures its meaning and relationships to other words.
        # Below we have a matrix of 65 by 32 as our vocab size is 65 characters long and 32 in embedding
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.blocks = nn.Sequential(
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            nn.LayerNorm(n_embed)
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)



    def forward(self,idx,targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tesnor of integers
        token_embed = self.token_embedding_table(idx) # B T C  MEANING BATCH TIME CHANNEL: batch: 4, block_size: 8, embed_size: 32
        pos_embed = self.position_embedding_table(torch.arange(T))  # T, C

        x = token_embed + pos_embed # (B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)


        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:] if idx.size(1) >= block_size else idx
            # get the predictions
            logits, loss = self(idx_cond) # this calls the forward function
            # foxus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the disstribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()


optimizer = torch.optim.AdamW(model.parameters(), learning_rate)


for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss: {losses['val']:.4f}")
    
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

print(decode(model.generate(idx= torch.zeros((1,1), dtype=torch.long), max_new_tokens=300)[0].tolist()))

