import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GeneralConv, GCNConv
from torch_geometric.nn.inits import glorot
import torch_geometric.utils as pyg_utils


_LAYER_UIDS = {}
keys_to_remove = [('gene', 'get_target', 'drug'), ('gene', 'interact', 'gene'),
                  ('drug', 'has_target', 'gene')]
gene_dim = 19081
drug_num = 645
drug_dim = 525

def reset_params(self):
    self.lin_msg.weight_initializer = "glorot"
    self.lin_msg.reset_parameters()

    if hasattr(self.lin_self, "reset_parameters"):
        self.lin_self.weight_initializer = "glorot"
        self.lin_self.reset_parameters()

    if not self.directed_msg:
        self.lin_msg_i.weight_initializer = "glorot"
        self.lin_msg_i.reset_parameters()

    if self.in_edge_channels is not None:
        self.lin_edge.weight_initializer = "glorot"
        self.lin_edge_weight.reset_parameters()

    if self.attention and self.attention_type == "additive":
        glorot(self.att_msg)

GeneralConv.reset_parameters = reset_params
def generate_hetero_conv_dict(hidden_dims, edge_types, num_layer):
    conv_dicts = []

    for i in range(num_layer):
        D = {}
        for edge_type in edge_types:
            src, relation, dst = edge_type
            D[edge_type] = GeneralConv((-1,-1), hidden_dims[i], aggr = "mean", skip_linear = 'True', l2_normalize = True)
        conv_dicts.append(D)
    return conv_dicts


class GraphConvolution(nn.Module):

    def __init__(self, input_dim, output_dim, device, act=None, dropout=0.):
        super(GraphConvolution, self).__init__()
        self.conv1 = GCNConv(input_dim, output_dim)
        self.dropout = dropout
        self.act = act if act is not None else nn.ReLU

    def forward(self, inputs, adj, reuse=False):
        if reuse:
            with torch.no_grad():
                x = F.dropout(inputs, p=self.dropout, training=self.training)
                x = self.conv1(x, adj)
                outputs = self.act(x)
        else:
            x = F.dropout(inputs, p=self.dropout, training=self.training)
            x = self.conv1(x, adj)
            outputs = self.act(x)
        return outputs


class GraphConvolutionK(nn.Module):

    def __init__(self, input_dim, output_dim, device, act=None, dropout=0.):
        super(GraphConvolutionK, self).__init__()
        self.conv1 = GCNConv(input_dim, output_dim)
        self.dropout = dropout
        self.act = act if act is not None else nn.ReLU

    def forward(self, inputs, adj, reuse=False):
        K = inputs.size(1)
        outputs = []
        if reuse:
            with torch.no_grad():
                for i in range(K):
                    x = F.dropout(inputs[:, i, :], p=self.dropout, training=self.training)
                    x = self.conv1(x, adj)
                    outputs.append(self.act(x).unsqueeze(1))
        else:
            for i in range(K):
                x = F.dropout(inputs[:, i, :], p=self.dropout, training=self.training)
                x = self.conv1(x, adj)
                outputs.append(self.act(x).unsqueeze(1))
        return torch.cat(outputs, dim=1)


class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""

    def __init__(self, act=None, dropout=0.):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act if act is not None else nn.Sigmoid()

    def forward(self, z, edge_index):
        src = z[edge_index[0]]  # N*(dim)16 N是边的数目，因为src为边的前一个节点的列表
        dst = z[edge_index[1]]


        out = (src * dst).sum(dim=1)
        return out



class SparseDecoder(nn.Module):
    def __init__(self, input_dim, dropout=0.):
        super(SparseDecoder, self).__init__()
        self.dropout = dropout
        self.rk = nn.Parameter(torch.rand(16))

    def forward(self, inputs):
        inputs = F.dropout(inputs, p=self.dropout, training=self.training)
        x = inputs.t()
        x = torch.mm(torch.diag(self.rk), x)
        x = torch.mm(inputs, x)
        outputs = 1 - torch.exp(-torch.exp(x))
        return outputs, self.rk


class GCNNModel(nn.Module):
    """Stack of graph convolutional layers."""

    def __init__(self, num_layers, output_dims, x_last_dim, device, relation, act=None, dropout=0.):
        super(GCNNModel, self).__init__()
        self.output_dims = output_dims
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = act if act is not None else nn.ReLU()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(
                    GraphConvolution(x_last_dim, output_dims[0], device, act, dropout))
            else:
                self.layers.append(
                    GraphConvolution(output_dims[i - 1] if i > 0 else output_dims[0], output_dims[i], device, act, dropout))


    def forward(self, inputs, adj, reuse=False):
        x = inputs
        for layer in self.layers:
            x = layer(x, adj, reuse)
        return x


class GCNNModelK(nn.Module):
    def __init__(self, num_layers, output_dims, noise_dim, x_last_dim, device,relation, act=None, dropout=0.):
        super(GCNNModelK, self).__init__()
        self.output_dims = output_dims
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = act if act is not None else nn.ReLU()
        self.layers = nn.ModuleList()
        self.noise_dim = noise_dim
        self.device = device

        for i in range(num_layers):
            if i == 0:
                self.layers.append(
                    GraphConvolutionK(x_last_dim, output_dims[0], device, act, dropout))
            else:
                if i < len(self.noise_dim):
                    self.layers.append(
                        GraphConvolutionK((output_dims[i - 1] + noise_dim[i]) if i > 0 else output_dims[0], output_dims[i], device, act, dropout))
                else:
                    self.layers.append(
                        GraphConvolutionK(output_dims[i - 1] if i > 0 else output_dims[0], output_dims[i], device, act, dropout))


    def forward(self, inputs, adj, K, is_add_noise=False, reuse=False):
        x = torch.tile(torch.unsqueeze(inputs, dim=1), [1, K, 1])
        for idx, layer in enumerate(self.layers):
            if is_add_noise and idx < len(self.noise_dim):
                B3 = BernoulliDistribution(0.5)

                e3 = B3.sample((x.size(0), K, self.noise_dim[idx])).float()
                e3 = torch.tensor(e3, device=self.device)

                input_ = torch.cat([e3, x], dim=2)

                x = layer(input_, adj, reuse)
            else:
                x = layer(x, adj, reuse)
        return x


class BernoulliDistribution(nn.Module):
    def __init__(self, prob):
        super(BernoulliDistribution, self).__init__()
        self.prob = prob

    def sample(self, shape):
        return torch.bernoulli(self.prob * torch.ones(shape))


'''--------------------------------------------------------------------------------------------------------------------------------------------'''


class SIPSEModel(nn.Module):
    def __init__(self, side_effect_edge_types, K, J, noise_dim, output_dim, hidden_dims, z_dim, augment_dim = 2048, dropout=0.1, device="cpu", eps=1e-10):
        super().__init__()

        self.K = K
        self.J = J
        self.device = device
        self.dropout = dropout
        self.noise_dim = noise_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.z_dim = z_dim
        # self.edge_types = edge_types
        self.side_effect_edge_types = side_effect_edge_types
        self.eps = eps
        self.augment_dim = augment_dim

        self.GCNNModel = nn.ModuleDict()
        self.GCNNModelK = nn.ModuleDict()
        self.GraphConvolution = nn.ModuleDict()
        self.GraphConvolutionK = nn.ModuleDict()
        self.linear_augment = nn.ModuleDict()
        # self.decoderMethod = nn.ModuleDict()
        output_dim_large = [256,128,32]
        for edge_type in side_effect_edge_types:
            src, relation, dst = edge_type
            self.GCNNModel[relation] = GCNNModel(len(output_dim), output_dim, self.hidden_dims[-1], device, relation, act=nn.ReLU(), dropout=dropout)
            self.GCNNModelK[relation] = GCNNModelK(len(output_dim), output_dim, noise_dim, self.hidden_dims[-1] + noise_dim[0], device, relation, act=nn.ReLU(), dropout=dropout)
            self.GraphConvolution[relation] = GraphConvolution(output_dim[-1], self.z_dim, device, act=nn.ReLU(), dropout=dropout)


            self.GraphConvolutionK[relation] = GraphConvolutionK(output_dim[-1], self.z_dim // 2, device, act=nn.ReLU(), dropout=dropout)
            self.linear_augment[relation] = nn.Linear(augment_dim, self.z_dim // 2).to(device)

        self.decoderMethod = InnerProductDecoder(act=nn.Sigmoid(), dropout=self.dropout)

    def reparameterize(self, mu, logstd):
        z_dict = {}
        for node_type in self.node_types:
            z_dict[node_type] = mu[node_type] + torch.randn_like(logstd[node_type]) * torch.exp(logstd[node_type])
        return z_dict

    def remove_gene(self, edge_index_dict):
        for key in keys_to_remove:
            edge_index_dict.pop(key, None)

    def encode(self, x_dict, edge_index_dict, augment):

        z_dict = x_dict.to_dense()


        z_sample_iws = {}
        log_H_iws = {}
        psi_iw_vecs = {}
        for key, edge_index in edge_index_dict.items():
            src, relation, dst = key

            z_logv = self.sample_logv(z_dict, edge_index, relation)
            z_logv_iw = torch.tile(torch.unsqueeze(z_logv, dim=1), [1, self.K, 1])
            sigma_iw1 = torch.exp(z_logv_iw / 2)
            sigma_iw2 = torch.tile(torch.unsqueeze(sigma_iw1, dim=2), [1, 1, self.J + 1, 1])

            psi_iw = self.sample_psi(z_dict, edge_index, self.K, relation)
            psi_iw = self.augment_latent(psi_iw, augment, relation, self.K)
            psi_iw_vec = torch.mean(psi_iw, dim=1)
            z_sample_iw = self.sample_n(psi_iw, sigma_iw1)
            z_sample_iws[relation] = z_sample_iw
            psi_iw_vecs[relation] = psi_iw_vec

            psi_iw_star = self.sample_psi(z_dict, edge_index, self.J, relation)
            psi_iw_star = self.augment_latent(psi_iw_star, augment, relation, self.J)
            psi_iw_star0 = torch.unsqueeze(psi_iw_star, dim=1)
            psi_iw_star1 = torch.tile(psi_iw_star0, [1, self.K, 1, 1])
            psi_iw_star2 = torch.cat([psi_iw_star1, torch.unsqueeze(psi_iw, dim=2)], 2)

            z_logv_iw = torch.tile(torch.unsqueeze(z_logv, dim=1), [1, self.K, 1])

            z_sample_iw1 = torch.unsqueeze(z_sample_iw, dim=2)
            z_sample_iw2 = torch.tile(z_sample_iw1, [1, 1, self.J + 1, 1])

            ker = torch.exp(
                -0.5 * torch.sum(torch.square(z_sample_iw2 - psi_iw_star2) / torch.square(sigma_iw2 + self.eps), 3))
            log_H_iw_vec = torch.log(torch.mean(ker, dim=2) + self.eps) - 0.5 * torch.sum(z_logv_iw, 2)
            log_H_iw = torch.mean(log_H_iw_vec, dim=0)
            log_H_iws[key] = log_H_iw

        return z_sample_iws, log_H_iws, psi_iw_vecs

    def get_z(self, item, relation):
        if relation == 'get_target':
            return item[:gene_dim, :]
        elif relation == 'has_target':
            return item[:drug_num, :]
        else:
            return item

    def cl_loss_iw(self, z_sample_iws, adjacency_sparse, warm_up, log_H_iws):

        loss_list = []
        for key, edge_index in adjacency_sparse.items():
            src, relation, dst = key
            log_lik_iws = []
            for i in range(self.K):
                z = z_sample_iws[relation][:,i,:]
                output = self.recon_loss(z, key, edge_index)
                log_lik_iws.append(output)

            log_lik_iw = torch.stack(log_lik_iws)

            log_prior_iw_vec = -0.5 * torch.sum(torch.square(z_sample_iws[relation]), 2)
            log_prior_iw = torch.mean(log_prior_iw_vec, dim=0)

            loss_iw0_relation = -torch.logsumexp((-log_lik_iw + (log_prior_iw - log_H_iws[key]) * warm_up / drug_num), dim=0, keepdim=True)

            loss_list.append(loss_iw0_relation)

        total_loss = torch.sum(torch.stack(loss_list))

        return total_loss

    def decode(self, z_dict, edge_index, edge_type, sigmoid = False):
        src, relation, dst = edge_type
        output = self.decoderMethod(z_dict, edge_index)
        if sigmoid:
            output = F.sigmoid(output)
        return output

    def decode_all_relation(self, z_dict, edge_index_dict, sigmoid=False):
        output = {}

        for edge_type in edge_index_dict.keys():
            src, relation, dst = edge_type
            output[relation] = self.decoderMethod(z_dict[relation], edge_index_dict[edge_type])
            if sigmoid:
                output[relation] = F.sigmoid(output[relation])
        return output

    def recon_loss(self, z_dict, edge_type, pos_edge_index, neg_edge_index = None):
        poss_loss = -torch.log(
                self.decode(z_dict, pos_edge_index, edge_type, sigmoid= True) + 1e-15).mean()

        if neg_edge_index is None:
            src, relation, dst = edge_type
            num_src_node, num_dst_node = z_dict.shape[0], z_dict.shape[0]
            neg_edge_index = pyg_utils.negative_sampling(pos_edge_index, num_nodes = (num_src_node, num_dst_node))

        neg_loss = -torch.log(1 -
                self.decode(z_dict, neg_edge_index, edge_type, sigmoid=True) + 1e-15).mean()

        return poss_loss + neg_loss

    # Define the sample_psi function
    def sample_psi(self, x, adjacency_sparse, K, relation, reuse=False):

        is_add_noise = True

        h3 = self.GCNNModelK[relation](x, adjacency_sparse, K, is_add_noise, reuse)

        mu = self.GraphConvolutionK[relation](h3, adjacency_sparse, reuse)

        return mu

    # Define the sample_logv function
    def sample_logv(self, x, adjacency_sparse, relation):
        net1 = self.GCNNModel[relation](x, adjacency_sparse)

        z_logv = self.GraphConvolution[relation](net1, adjacency_sparse)

        return z_logv

    # Define the sample_n function
    def sample_n(self, psi, sigma):
        eps = torch.randn(psi.size(), device=self.device)
        z = psi + eps * sigma
        return z

    def augment_latent(self, z_sample_iw, augment, relation, K):
        temp_augment = self.linear_augment[relation](augment)
        temp_augment = torch.tile(torch.unsqueeze(temp_augment, dim=1), [1, K, 1])
        z_sample_iw = torch.cat([z_sample_iw, temp_augment], dim = -1)
        return z_sample_iw
