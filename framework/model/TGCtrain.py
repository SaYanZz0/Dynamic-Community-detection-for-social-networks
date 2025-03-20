import math

import torch
from torch.autograd import Variable
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from sklearn.cluster import KMeans
import numpy as np
import sys
from model.DataSet import TGCDataSet
from model.evaluation import eva
from torch.nn import Linear
import torch.nn.functional as F

# Define tensor types for convenience
FType = torch.FloatTensor
LType = torch.LongTensor

DID = 0


class TGC:
    """
    Temporal Graph Clustering (TGC) model implementation.
    This model learns temporal-aware node embeddings and performs clustering on them.
    """
    def __init__(self, args):
        """
        Initialize the TGC model with configuration parameters.
        
        Args:
            args: Configuration parameters for the model
        """
        self.args = args
        self.the_data = args.dataset  # Dataset name
        
        # Define file paths for data, embeddings, and labels
        self.file_path = '../data/%s/%s.txt' % (self.the_data, self.the_data)  # Path to temporal graph edges
        self.emb_path = '../emb/%s/%s_TGC_%d.emb'  # Path template for saving embeddings
        self.feature_path = './pretrain/%s_feature.emb' % self.the_data  # Path to pretrained node features
        self.label_path = '../data/%s/node2label.txt' % self.the_data  # Path to ground truth node labels
        self.labels = self.read_label()  # Load ground truth labels
        
        # Model hyperparameters
        self.emb_size = args.emb_size  # Embedding dimension
        self.neg_size = args.neg_size  # Number of negative samples
        self.hist_len = args.hist_len  # History length (temporal context)
        self.batch = args.batch_size  # Batch size for training
        self.clusters = args.clusters  # Number of clusters to form
        self.save_step = args.save_step  # Frequency of saving embeddings
        self.epochs = args.epoch  # Number of training epochs
        
        # Performance tracking
        self.best_acc = 0
        self.best_nmi = 0
        self.best_ari = 0
        self.best_f1 = 0
        
        # Early stopping parameters
        self.patience = args.patience if hasattr(args, 'patience') else 1  # Wait for n epochs before early stopping
        self.min_delta = args.min_delta if hasattr(args, 'min_delta') else 0.1  # Minimum change to count as improvement

        # Load dataset and initialize model components
        self.data = TGCDataSet(self.file_path, self.neg_size, self.hist_len, self.feature_path, args.directed)
        self.node_dim = self.data.get_node_dim()  # Number of nodes in the graph
        self.edge_num = self.data.get_edge_num()  # Number of edges in the graph
        self.feature = self.data.get_feature()  # Get pretrained node features
        
        # Initialize model parameters
       
        self.node_emb = Variable(torch.from_numpy(self.feature).type(FType), requires_grad=True)
        self.pre_emb = Variable(torch.from_numpy(self.feature).type(FType).cpu(), requires_grad=False)
        # Delta parameter for temporal attention
        self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cpu(), requires_grad=True)

        # Initialize cluster centers
        self.cluster_layer = Variable((torch.zeros(self.clusters, self.emb_size) + 1.).type(FType).cpu(), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)  
        
        # Use KMeans to initialize cluster centers based on node features
        kmeans = KMeans(n_clusters=self.clusters, n_init=20)
        _ = kmeans.fit_predict(self.feature)
        self.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cpu()
        
        # Additional model parameters
        self.v = 1.0  # Parameter for soft assignment formula
        self.batch_weight = math.ceil(self.batch / self.edge_num)  # Weight for loss normalization
        
        # Optimizer and loss initialization
        self.opt = SGD(lr=args.learning_rate, params=[self.node_emb, self.delta, self.cluster_layer])
        self.loss = torch.FloatTensor()

    def read_label(self):
        """
        Read ground truth cluster labels from file.
        
        Returns:
            List of integer labels for each node.
        """
        n2l = dict()
        labels = []
        with open(self.label_path, 'r') as reader:
            for line in reader:
                parts = line.strip().split()
                n_id, l_id = int(parts[0]), int(parts[1])
                n2l[n_id] = l_id
        reader.close()
        for i in range(len(n2l)):
            labels.append(int(n2l[i]))
        return labels

    def kl_loss(self, z, p):
        """
        Calculate KL divergence loss between cluster assignments and target distribution.
        
        Args:
            z: Current node embeddings [batch_size, emb_size]
            p: Target distribution [batch_size, clusters]
            
        Returns:
            KL divergence loss between learned cluster assignments and target distribution
        """
        # Calculate pairwise squared distances between nodes and cluster centers
        # z.unsqueeze(1) creates [batch_size, 1, emb_size] for broadcasting
        # Result is [batch_size, clusters] tensor of distances
        distances = torch.pow(z.unsqueeze(1) - self.cluster_layer, 2)
        
        # Convert distances to probabilities using Student's t-distribution (soft assignment)
        q = 1.0 / (1.0 + torch.sum(distances, 2) / self.v)  # [batch_size, clusters]
        q = q.pow((self.v + 1.0) / 2.0)  # Raise to power of (v+1)/2 for distribution sharpening
        q = (q.t() / torch.sum(q, 1)).t()  # Normalize so each row sums to 1

        # Calculate KL divergence between learned distribution q and target distribution p
        # KL divergence requires log probabilities for input
        the_kl_loss = F.kl_div(q.log(), p, reduction='batchmean')  # l_clu
        return the_kl_loss

    def target_dis(self, emb):
        """
        Compute target distribution P for self-training clustering.
        
        Args:
            emb: Node embeddings [batch_size, emb_size]
            
        Returns:
            Target distribution p that emphasizes high-confidence assignments
        """
        # Calculate initial soft assignments (same as in kl_loss)
        distances = torch.pow(emb.unsqueeze(1) - self.cluster_layer, 2)
        q = 1.0 / (1.0 + torch.sum(distances, 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()  # Normalized Q matrix [batch_size, clusters]

        # Refine target distribution:
        tmp_q = q.data  # Use detached version for target calculation
        weight = tmp_q ** 2 / tmp_q.sum(0)  # Emphasize high-confidence assignments
        p = (weight.t() / weight.sum(1)).t()  # Normalize again to get probability distribution

        return p

    def forward(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        """Forward pass of the TGC model combining temporal, structural and clustering objectives.
        
        Args:
            s_nodes: Source nodes [batch_size]
            t_nodes: Target nodes [batch_size]
            t_times: Target timestamps [batch_size]
            n_nodes: Negative samples [batch_size, neg_size]
            h_nodes: Historical context nodes [batch_size, hist_len]
            h_times: Historical timestamps [batch_size, hist_len]
            h_time_mask: Mask for valid historical entries [batch_size, hist_len]
            
        Returns:
            total_loss: Combined loss value for optimization
        """
        # Get batch size and retrieve embeddings for all node types
        batch = s_nodes.size()[0]
        s_node_emb = self.node_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)  # [batch_size, emb_size]
        t_node_emb = self.node_emb.index_select(0, Variable(t_nodes.view(-1))).view(batch, -1)  # [batch_size, emb_size]
        h_node_emb = self.node_emb.index_select(0, Variable(h_nodes.view(-1))).view(batch, self.hist_len, -1)  # [batch_size, hist_len, emb_size]
        n_node_emb = self.node_emb.index_select(0, Variable(n_nodes.view(-1))).view(batch, self.neg_size, -1)  # [batch_size, neg_size, emb_size]
        s_pre_emb = self.pre_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)  # [batch_size, emb_size]

        # Clustering component losses
        s_p = self.target_dis(s_pre_emb)  # Get target distribution for clustering
        s_kl_loss = self.kl_loss(s_node_emb, s_p)  # KL divergence loss for cluster assignments
        l_node = s_kl_loss  # Node-level clustering loss

        # Structural preservation losses
        # Source-target similarity loss
        new_st_adj = torch.cosine_similarity(s_node_emb, t_node_emb)  # [batch_size]
        res_st_loss = torch.norm(1 - new_st_adj, p=2, dim=0)  # L2 norm of similarity difference
        
        # Source-history similarity loss
        new_sh_adj = torch.cosine_similarity(s_node_emb.unsqueeze(1), h_node_emb, dim=2)  # [batch_size, hist_len]
        new_sh_adj = new_sh_adj * h_time_mask  # Apply time-aware masking
        res_sh_loss = torch.norm(1 - new_sh_adj, p=2, dim=0).sum(dim=0, keepdims=False)  # Masked similarity loss
        
        # Source-negative similarity loss
        new_sn_adj = torch.cosine_similarity(s_node_emb.unsqueeze(1), n_node_emb, dim=2)  # [batch_size, neg_size]
        res_sn_loss = torch.norm(0 - new_sn_adj, p=2, dim=0).sum(dim=0, keepdims=False)  # Push negatives apart
        
        l_batch = res_st_loss + res_sh_loss + res_sn_loss  # Combine structural losses
        l_framework = l_node + l_batch  # Total framework loss (clustering + structure)

        # Temporal attention calculation
        # Compute attention weights based on squared distance between current and historical embeddings
        att = softmax(((s_node_emb.unsqueeze(1) - h_node_emb) ** 2).sum(dim=2).neg(), dim=1)

        # Positive pair scoring components
        p_mu = ((s_node_emb - t_node_emb) ** 2).sum(dim=1).neg()  # Direct source-target similarity
        p_alpha = ((h_node_emb - t_node_emb.unsqueeze(1)) ** 2).sum(dim=2).neg()  # Historical-target similarity
        
        # Time decay calculation
        delta = self.delta.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)  # Node-specific decay rates
        d_time = torch.abs(t_times.unsqueeze(1) - h_times)  # Time differences
        # Combined positive score with temporal attention
        p_lambda = p_mu + (att * p_alpha * torch.exp(delta * Variable(d_time)) * Variable(h_time_mask)).sum(dim=1)

        # Negative pair scoring components
        n_mu = ((s_node_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg()  # Source-negative similarity
        n_alpha = ((h_node_emb.unsqueeze(2) - n_node_emb.unsqueeze(1)) ** 2).sum(dim=3).neg()  # Historical-negative similarity
        
        # Combined negative score with temporal attention
        n_lambda = n_mu + (att.unsqueeze(2) * n_alpha * (torch.exp(delta * Variable(d_time)).unsqueeze(2)) * (
            Variable(h_time_mask).unsqueeze(2))).sum(dim=1)

        # Final loss calculation
        loss = -torch.log(p_lambda.sigmoid() + 1e-6) - torch.log(n_lambda.neg().sigmoid() + 1e-6).sum(dim=1)
        total_loss = loss.sum() + l_framework  # Combine all loss components

        return total_loss

    def update(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        """
        Update model parameters based on a batch of data.
        
        Args:
            s_nodes: Source nodes
            t_nodes: Target nodes
            t_times: Timestamps of target nodes
            n_nodes: Negative samples
            h_nodes: Historical context nodes
            h_times: Timestamps of historical nodes
            h_time_mask: Mask for valid historical interactions
            
        This method performs a single optimization step for the model.
        """
        # Perform optimization step with accumulated gradients if we have enough samples
        if self.mini_batch > 0:
            self.mini_batch -= 1
            loss = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
            self.loss += loss.data
            loss.backward()
            if self.mini_batch == 0:
                self.opt.step()
        else:
            # Regular optimization step
            self.opt.zero_grad()
            loss = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
            self.loss += loss.data
            loss.backward()

    def train(self):
        """
        Train the TGC model for the specified number of epochs.
        
        This method:
        1. Loads data in batches
        2. Performs model updates
        3. Evaluates clustering performance
        4. Implements early stopping
        5. Saves the best model
        """
        
        epochs_no_improve = 0
        best_epoch = 0
        last_nmi = 0
        
        for epoch in range(self.epochs):
            self.loss = 0.0
            # Create data loader for batching
            loader = DataLoader(self.data, batch_size=self.batch, shuffle=True, num_workers=0)

            # Process each batch
            for i_batch, sample_batched in enumerate(loader):
                if i_batch != 0:
                    sys.stdout.write('\r' + str(i_batch * self.batch) + '\tloss: ' + str(
                        self.loss.cpu().numpy() / (self.batch * i_batch)))
                    sys.stdout.flush()

                # Update model parameters with current batch
                self.update(sample_batched['source_node'].type(LType).cpu(),
                            sample_batched['target_node'].type(LType).cpu(),
                            sample_batched['target_time'].type(FType).cpu(),
                            sample_batched['neg_nodes'].type(LType).cpu(),
                            sample_batched['history_nodes'].type(LType).cpu(),
                            sample_batched['history_times'].type(FType).cpu(),
                            sample_batched['history_masks'].type(FType).cpu())

            # Skip evaluation for large datasets (known to be computationally expensive)
            if self.the_data == 'arxivLarge' or self.the_data == 'arxivPhy' or self.the_data == 'arxivMath':
                acc, nmi, ari, f1 = 0, 0, 0, 0
            else:
                # Evaluate clustering performance
                acc, nmi, ari, f1 = eva(self.clusters, self.labels, self.node_emb)

            # Handle best model saving and early stopping tracking
            if nmi > self.best_nmi:
                # We found a better model - update best metrics
                self.best_acc = acc
                self.best_nmi = nmi
                self.best_ari = ari
                self.best_f1 = f1
                best_epoch = epoch
                # Save the best embeddings
                self.save_node_embeddings(self.emb_path % (self.the_data, self.the_data, self.epochs))

            else:
                # No improvement in this epoch
                epochs_no_improve += 1

            # Print progress
            sys.stdout.write('\repoch %d: loss=%.4f  ' % (epoch, (self.loss.cpu().numpy() / len(self.data))))
            sys.stdout.write('ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f) ' % (acc, nmi, ari, f1))
            sys.stdout.write('Best NMI(%.4f) at epoch %d\n' % (self.best_nmi, best_epoch))
            sys.stdout.flush()
            
            # Early stopping check - only after a minimum number of epochs
            if epochs_no_improve >= self.patience and epoch > 20:
                sys.stdout.write('\rEarly stopping triggered. No improvement for %d epochs (best NMI: %.4f at epoch %d)\n' 
                                % (self.patience, self.best_nmi, best_epoch))
                break

        # Print final results
        print('Best performance: ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f) at epoch %d' %
              (self.best_acc, self.best_nmi, self.best_ari, self.best_f1, best_epoch))
        

    def save_node_embeddings(self, path):
        """
        Save learned node embeddings to a file.
        
        Args:
            path: Path to save the embeddings
        
        The saved format includes:
        - First line: <num_nodes> <embedding_size>
        - Subsequent lines: <node_id> <embedding_vector>
        """
        if torch.cuda.is_available():
            embeddings = self.node_emb.cpu().data.numpy()
        else:
            embeddings = self.node_emb.data.numpy()
        writer = open(path, 'w')
        writer.write('%d %d\n' % (self.node_dim, self.emb_size))
        for n_idx in range(self.node_dim):
            writer.write(str(n_idx) + ' ' + ' '.join(str(d) for d in embeddings[n_idx]) + '\n')

        writer.close()
