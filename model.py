import torch
import torch.nn as nn
import keras
from keras.layers import Dense
import tensorflow as tf
class Conv1dUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, padding=0, batchnorm=True,
                 nonlinearity=nn.LeakyReLU(0.2)):
        super(Conv1dUnit, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                    nn.Conv1d(in_channels, 10, kernel, stride, padding),
                    nn.BatchNorm1d(out_channels), nonlinearity)
        else:
            self.model = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel, stride, padding), nonlinearity)

    def forward(self, x):
        return self.model(x)

class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, batchnorm=True, nonlinearity=nn.LeakyReLU(0.2)):
        super(LinearUnit, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features), nonlinearity)
        else:
            self.model = nn.Sequential(
                    nn.Linear(in_features, out_features), nonlinearity)

    def forward(self, x):
        return self.model(x)


class DisentangledEEG(nn.Module):
    """
    Network Architecture:
    Z prior:
        The z prior is Gaussian (This makes sense, since aside from reparameterization convenience,
         , z encodes time variant features/artifcats/noices across EEG signals)
         Mean and variance computed by LSTM as follows:
         h_t, c_t = prior_lstm(z_t, (h_t-1,c_t-1)) where h_t and c_t are hidden state and cell state respectively.
         The hidden_state h_t is used to compute to mean and variance of z_t using and affine transformation:
         z_mean, z_log_variance = affine_mean(h_t), affine_logvar(h_t)
         z = reparameterize(z_mean, z_log_variance)
         The hidden state has dimension hidden_dim and z has dimension z_dim (32 by default_.
    Approximate Posterior of f (which encodes time-invariant features):
        The approximate posterior is parameterized by a bidirectional LSTM that takes the entire sequence of transformed
        x_t (transformation is being done by Conv1d unit which is essentially a fully connected matrix to encode all
         the EEG channels conveniently into a code_dim) as input at each time step.
          he hidden layer dimension is hidden_dim.
         The features from the unit corresponding to the last time step of the forward LSTM and the unit corresponding
         to the first time step of backward LSTM (as in the paper) are concatenated and fed into two affine layers
         (without added nonlinearity) to compute the mean and variance of the gaussian posterior of f.
    Approximate posterior for z (factored q (similat to an amortized inference))
        Each x_t is first fed into an affine layer followed by a LeakyRelU(0.2) to generate and intermediate feature
        vector of dimension hidden_dim, which is then followed 2 affine layers (without non-linearity) to compute
         the mean and variance of the Gaussian posterior of each z_t.
         inter_t = intermediate(x_t)
         z_mean_t , z_log_variance = affine_mean(inter_t), affine_logvar(inter_t)
         z = reparameterize(z_mean_t, z_log_variance_t)
    Approximate posterior for z (Full q)
    The vector f is concatenated to each v_t where v_t is the encodings for each eeg measurement x_t,
     by the conv1d encoder. This entire sequence is fed into a bi-LSTM of hidden layer of dimension hidden_dim.
     The output h_t of each time step of this RNN transformed by two affine transformation
     (reparemeterization alwasy without nonlinearity) to compute the mean and variance of the gaussian posterior of
     each z_t as the following:
     g_t = [v_t, f] for each timestep
     forward_features, backward_features = lstm(g_t for all time steps)
     h_t = rnn([forward_features, backward])
     z_mean_t, z_log_variance_t = affine_mean(h_t), affine_logvar(h_t)
     z = reparemeterize(z_mean, z_logvar_t)
     Decoder for conditional distributions p(x_t|f,z_t)
        The architecture is symmetric to that of conv1d encoder. The vector f is concatenated to each z_t, which
        the goes through affine transformations, batchnorm and leakyrelu(0.2).
    Hyperparameters:
        f_dim: dimension of the time-invariant encodings, with the shape (batch_size, f_fim)
        z_dim: dimension of the time varying encodings, with the shape (batch_size, seq_len, z_dim)
        seq_len: Number of the time steps in the convolved eeg signals , in our case
        the orig seq_lenght is 3518, after conv1d with kernel=20, stride=5 reduces to 350, in the range park the
        of a practical lstm.
        hidden_dim: Dimension of the LSTM/RNNs.
        nonlinearity: Parameterized LeakyRelu(0.2)
        num_channels: number of channels in eeg measurements
        conv_dim: The convolutional encoder converts each timesep into an intermediate encoding vector of size conv_dim, i.e,
                  The initial eeg tensor (batch_size, seq_len=3518, num_channels, in_size, in_size) is converted to
                  (batch_size, seq_len=350, conv_dim)
        factorized: to switch between the factorized and full z  posterior approximation.
    Optimization:
        Adam
    """
    def __init__(self, f_dim=256, z_dim=32, in_size=2, channels=40, conv_dim=10, hidden_dim=256, seq_len=1,
                 factorized=True, nonlinearity=True, kernel=345, stride=160, target_size=1 ):
        super(DisentangledEEG, self).__init__()
        print(f_dim,'f_dim')
        print(z_dim,'z_dim')
        self.f_dim = f_dim
        self.z_dim = z_dim
        self.seq_len = seq_len
        self.conv_dim = conv_dim
        self.hidden_dim = hidden_dim
        self.factorized = factorized
        self.channels = channels
        self.in_size = in_size
        self.kernel = kernel
        self.stride = stride
        self.target_size = target_size

        # For now we consider the prior of Beat EPRS (event_related_potentials) to be Gaussian.
        # An ERP is an electrophysiological response that occurs as a direct result of a stimulus.
        # TODO:  In the current MIIR case is an audio stumili.
        #  Interestingly, in both cases of EEG signal and auditory data, prior over the time invariant encoder f
        #  are better to be distributions
        #  with higher kurtosis than the Gaussian such as Laplacian.
        #  However, we may want to consider the time-variant features to be still Gaussian.
        self.z_prior_lstm = nn.LSTMCell(self.z_dim, self.hidden_dim)
        self.z_prior_mean = nn.Linear(self.hidden_dim, z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, z_dim)
        self.f_lstm = nn.LSTM(self.conv_dim, self.hidden_dim, 1 , bidirectional=True, batch_first=True)
        self.f_mean = LinearUnit(self.hidden_dim*2, self.f_dim, False)
        self.f_logvar = LinearUnit(self.hidden_dim*2, self.f_dim, False)

        if self.factorized is True:
            self.z_inter = LinearUnit(self.conv_dim, self.hidden_dim, batchnorm=False)
            self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
            self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        else:
            self.z_lstm = nn.LSTM(self.conv_dim + self.f_dim, self.hidden_dim, 1, bidirectional=True,
                                  batch_first=True)
            self.z_rnn = nn.RNN(self.hidden_dim*2, self.hidden_dim, batch_first=True)
            # Each timestep is for each z so no reshaping and feature mixing
            self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
            self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)
        #print(channels)
        self.conv = nn.Sequential(Conv1dUnit(channels, conv_dim, kernel=kernel, stride=stride),
                                  nn.BatchNorm1d(conv_dim))
        # NOTE: Targets are better classified using z_out (i.e encoders of time invariant features).
        self.target = nn.Sequential(nn.Linear(f_dim, target_size), nn.Tanh())

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def sample_z(self, batch_size, random_sampling = True):
        z_out = None # This will ultimately store all z_s in the format [batch_size, len_seq, z_dim]
        z_means = None
        z_logvars = None

        # All states are initially set to 0, especially z_0 = 0
        z_t = torch.zeros(batch_size, self.z_dim)
        z_mean_t = torch.zeros(batch_size, self.z_dim)
        z_logvar_t = torch.zeros(batch_size, self.z_dim)
        h_t = torch.zeros(batch_size, self.hidden_dim)
        c_t = torch.zeros(batch_size, self.hidden_dim)

        for _ in range(self.seq_len):
            h_t, c_t = self.z_prior_lstm(z_t, (h_t, c_t))
            z_mean_t = self.z_prior_mean(h_t)
            z_logvar_t = self.z_prior_logvar(h_t)
            z_t = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store in the format [batch_size, 1, z_dim]
                z_out = z_t.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars  = z_logvar_t.unsqueeze(1)
            else:
                # if z_out is not none, append all the sequential collected z_t in z_out
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
        return z_means, z_logvars, z_out

    def encode_signals(self, x):
        #print(x.shape,'waybefore')
        #print(x[0])
        
        x = x.squeeze(-1)
        #print(x.shape,'before')
        x = self.conv(x)
        #print(x.shape,'after')
        x = x.view(-1, self.seq_len, self.conv_dim)
        #print(x.shape,'final')
        return x

    def reparameterize(self, mean, logvar, random_sampling = False):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.randn_like(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean

    def encode_f(self, x):
        lstm_out, _ = self.f_lstm(x)
        # The features of the last timestep of the forward RNN is stored at the end of lstm_out in the first half,
        # and the features of the "first timestep" of the backward RNN is stored at the beginning of lstm_out in the
        # second half.
        # For a detailed explanation, check https://gist.github.com/ceshine/bed2dadca48fe4fe4b4600ccce2fd6e1
        backward = lstm_out[:, 0, self.hidden_dim:2*self.hidden_dim]
        frontal = lstm_out[:, self.seq_len-1, 0:self.hidden_dim]
        lstm_out = torch.cat((frontal, backward), dim=1)
        mean = self.f_mean(lstm_out)
        logvar = self.f_logvar(lstm_out)
        # self.training so during testing no sampling would be used.
        return mean, logvar, self.reparameterize(mean, logvar, False)

    def encode_z(self, x, f):
        if self.factorized is True:
            features = self.z_inter(x)
        else:
            # The expansion is done to match the dimension of the x and f, used for concatenating f for each x_t
            f_expand = f.unsqueeze(1).expand(-1, self.seq_len, self.f_dim)
            lstm_out, _ = self.z_lstm(torch.cat((x, f_expand), dim = 2))
            features, _ = self.z_rnn(lstm_out)

        mean = self.z_mean(features)
        logvar = self.z_logvar(features)
        return mean, logvar, self.reparameterize(mean, logvar, False)

    def forward(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=False)
        conv_x = self.encode_signals(x)
        #conv_x = x
        f_mean, f_logvar, f = self.encode_f(conv_x)
        z_mean, z_logvar, z = self.encode_z(conv_x, f)
        f_expand = f.unsqueeze(1).expand(-1, self.seq_len, self.f_dim)
        zf = torch.cat((z, f_expand), dim=2)
        # :Note replace z with zf or f, would lower the accuracy.
        # target_pred = self.target(z.view(z.size(1),z.size(0),-1)[-1])
        # target_pred = self.target(zf.view(z.size(1), zf.size(0), -1)[-1])
        target_pred = self.target(f)
        return f_mean, f_logvar, f, z_mean, z_logvar, z, z_mean_prior, z_logvar_prior, target_pred