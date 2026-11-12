import torch
import torch.nn as nn


class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """
    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))

class Generator(nn.Module):
    def __init__(self,  modality, device,
                 input_dim=11, embedding=False, latent_layer_idx=-1,
                 hidden_dim=512, latent_dim=816, num_classes=None, ):
        super(Generator, self).__init__()
        self.embedding = embedding
        #self.model=
        self.device = device
        self.latent_layer_idx = latent_layer_idx
        self.modality =modality
        # if len(self.modality)!=0:
        #     if len(self.modality)==1:
        #         if self.modality==[1]:
        #             self.hidden_dim, self.latent_dim, self.input_channel, self.n_class, self.noise_dim = 512,576,1,11,11
        #         if self.modality==[2]:
        #             self.hidden_dim, self.latent_dim, self.input_channel, self.n_class, self.noise_dim = 512,560,1,11,11
        #         if self.modality==[3]:
        #             self.hidden_dim, self.latent_dim, self.input_channel, self.n_class, self.noise_dim = 512,496,1,11,11
        #     elif len(self.modality)==2:
        #         if self.modality==[1,2]:
        #             self.hidden_dim, self.latent_dim, self.input_channel, self.n_class, self.noise_dim = 512,320,1,11,11
        #         if self.modality==[1,3]:
        #             self.hidden_dim, self.latent_dim, self.input_channel, self.n_class, self.noise_dim = 512,256,1,11,11
        #         if self.modality == [2, 3]:
        #             self.hidden_dim, self.latent_dim, self.input_channel, self.n_class, self.noise_dim = 512, 240, 1, 11, 11
        # else:
        self.hidden_dim, self.latent_dim, self.input_channel, self.n_class, self.noise_dim = hidden_dim, latent_dim, None, num_classes, num_classes
        self.fc_configs = [input_dim*2, self.hidden_dim]
        self.init_loss_fn()
        self.build_network()

    def get_number_of_parameters(self):
        pytorch_total_params=sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params

    def init_loss_fn(self):
        self.crossentropy_loss=nn.NLLLoss(reduce=False) # same as above
        self.diversity_loss = DiversityLoss(metric='l1')
        self.dist_loss = nn.MSELoss()

    def build_network(self):
        if self.embedding:
            self.embedding_layer = nn.Embedding(self.n_class, self.noise_dim)
        ### FC modules ####
        self.fc_layers = nn.ModuleList()
        for i in range(len(self.fc_configs) - 1):
            input_dim, out_dim = self.fc_configs[i], self.fc_configs[i + 1]
            fc = nn.Linear(input_dim, out_dim)
            bn = nn.BatchNorm1d(out_dim)
            act = nn.ReLU()
            self.fc_layers += [fc, bn, act]
        ### Representation layer
        self.representation_layer = nn.Linear(self.fc_configs[-1], self.latent_dim)
        # print("Build last layer {} X {}".format(self.fc_configs[-1], self.latent_dim))

    def forward(self, labels, latent_layer_idx=-1, verbose=True):
        """
        G(Z|y) or G(X|y):
        Generate either latent representation( latent_layer_idx < 0) or raw image (latent_layer_idx=0) conditional on labels.
        :param labels:
        :param latent_layer_idx:
            if -1, generate latent representation of the last layer,
            -2 for the 2nd to last layer, 0 for raw images.
        :param verbose: also return the sampled Gaussian noise if verbose = True
        :return: a dictionary of output information.
        """
        labels.cuda(self.device)
        # print(labels)
        result = {}
        batch_size = labels.shape[0]
        eps = torch.rand((batch_size, self.noise_dim)).cuda(self.device)# sampling from Gaussian
        if verbose:
            result['eps'] = eps
        if self.embedding: # embedded dense vector
            y_input = self.embedding_layer(labels).cuda(self.device)
        else: # one-hot (sparse) vector
            y_input = torch.FloatTensor(batch_size, self.n_class).cuda(self.device)
            y_input.zero_()
            #labels = labels.view
            y_input.scatter_(1, labels.view(-1,1), 1)
            # print(y_input)

        z = torch.cat((eps, y_input), dim=1)
        ### FC layers
        for layer in self.fc_layers:
            z = layer(z)
        z = self.representation_layer(z)
        result['output'] = z
        return result