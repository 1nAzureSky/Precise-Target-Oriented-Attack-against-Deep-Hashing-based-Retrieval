import torch

class SmoothAP(torch.nn.Module):
    def __init__(self, anneal):
        """
        Parameters
        ----------
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function
        """
        super(SmoothAP, self).__init__()
        self.anneal = anneal

    def sigmoid(self, tensor, temp=0.1):
        """
        temperature controlled sigmoid
        takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
        """
        exponent = -tensor / temp
        # clamp the input tensor for stability
        exponent = torch.clamp(exponent, min=-50, max=50)
        y = 1.0 / (1.0 + torch.exp(exponent))
        return y

    def hmm(self, x, y, k):
        """
        Parameters
        ----------
        x : query (bit)
        y : retrieval set (batch,bit)
        k : bit
        """
        return ((k - torch.mm(x, y.permute(1, 0))) / 2)

    def caculate_rank(self, x, y):  # x.shape:(bit),y.shape:(batch,bit)
        k = len(x)
        batch = y.shape[0]
        mask = (1 - torch.eye(batch)).cuda()

        query_sim = self.hmm(x, y, k)
        query_D = (query_sim.repeat(batch, 1)) - (query_sim.permute(1, 0).repeat(1, batch))
        sim_sg = (self.sigmoid(query_D, temp=self.anneal) * mask).cuda()
        all_rk = torch.sum(sim_sg, dim=0) + 1
        return all_rk

    def forward(self, query, database, pos_len):
        all_rank = self.caculate_rank(query, database)
        pos_rank = self.caculate_rank(query, database[:pos_len])
        AP = torch.sum(pos_rank / all_rank[:len(pos_rank)]) / len(pos_rank)
        # print(AP.data)

        return (1 - AP)