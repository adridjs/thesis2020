import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
        self.word2idx = {}
        self.epochs = None
        self.ngrams = []

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


def train(model):
    for epoch in range(10):
        total_loss = 0
        losses = []
        for context, target in model.ngrams:
            # Prepare the inputs to be passed to the model
            context_idxs = torch.tensor([model.word2idx[w] for w in context], dtype=torch.long)
            model.zero_grad()
            log_probs = model(context_idxs)
            loss = loss_function(log_probs, torch.tensor([model.word2idx[target]], dtype=torch.long))
            loss.backward()
            optimizer.step()

            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
            losses.append(total_loss)

        print(losses)


if __name__ == '__main__':
    lr = 10e-3
    model = CBOW(len([]), embedding_dim=512, context_size=5)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train(model)

    # TODO: argparser


