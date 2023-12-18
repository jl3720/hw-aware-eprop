import numpy as np
import torch
from collections import namedtuple

import matplotlib.pyplot as plt


def sum_of_sines_target(seq_len, n_sines=4, periods=[1000, 500, 333, 200], weights=None, phases=None, normalize=True):
    '''
    Generate a target signal as a weighted sum of sinusoids with random weights and phases.
    :param n_sines: number of sinusoids to combine
    :param periods: list of sinusoid periods
    :param weights: weight assigned the sinusoids
    :param phases: phases of the sinusoids
    :return: one dimensional vector of size seq_len contained the weighted sum of sinusoids
    '''
    if periods is None:
        periods = [np.random.uniform(low=100, high=1000) for i in range(n_sines)]
    assert n_sines == len(periods)
    sines = []
    weights = np.random.uniform(low=0.5, high=2, size=n_sines) if weights is None else weights
    phases = np.random.uniform(low=0., high=np.pi * 2, size=n_sines) if phases is None else phases
    for i in range(n_sines):
        sine = np.sin(np.linspace(0 + phases[i], np.pi * 2 * (seq_len // periods[i]) + phases[i], seq_len))
        sines.append(sine * weights[i])

    output = sum(sines)
    if normalize:
        output = output - output[0]
        scale = max(np.abs(np.min(output)), np.abs(np.max(output)))
        output = output / np.maximum(scale, 1e-6)
    return output
    
    
def pseudo_derivative(v_scaled, dampening_factor):
    '''
    Define the pseudo derivative used to derive through spikes.
    :param v_scaled: scaled version of the voltage being 0 at threshold and -1 at rest
    :param dampening_factor: parameter that stabilizes learning
    :return:
    '''
    return torch.maximum(1 - torch.abs(v_scaled), torch.tensor(0)) * dampening_factor

class SpikeFunction_class(torch.autograd.Function):
    dampening_factor = 0.3
    @staticmethod
    def forward(ctx, input, thr):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > thr] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        dE_dz = grad_output.clone()
        #grad = grad_input/(torch.abs(sgt_heaviside.scale*(input - thr))+1.0)**2
        dE_dz_scaled = torch.maximum(1 - torch.abs(input), torch.tensor([0])) * 0.3
        return dE_dz*dE_dz_scaled, None
    
SpikeFunction = SpikeFunction_class.apply

LightLIFStateTuple = namedtuple('LightLIFStateTuple', ('v', 'z'))
class LightLIF(torch.nn.Module):
    def __init__(self, n_in=4, n_rec=4, tau=20., thr=0.615, dt=1., dtype=torch.float32, dampening_factor=0.3,
                 stop_z_gradients=False):
        super(LightLIF, self).__init__()
        '''
        A tensorflow RNN cell model to simulate Learky Integrate and Fire (LIF) neurons.

        WARNING: This model might not be compatible with tensorflow framework extensions because the input and recurrent
        weights are defined with tf.Variable at creation of the cell instead of using variable scopes.

        :param n_in: number of input neurons
        :param n_rec: number of recurrenet neurons
        :param tau: membrane time constant
        :param thr: threshold voltage
        :param dt: time step
        :param dtype: data type
        :param dampening_factor: parameter to stabilize learning
        :param stop_z_gradients: if true, some gradients are stopped to get an equivalence between eprop and bptt
        '''

        self.dampening_factor = dampening_factor
        self.dt = float(dt)
        self.n_in = n_in
        self.n_rec = n_rec
        self.data_type = dtype
        self.stop_z_gradients = stop_z_gradients

        self._num_units = self.n_rec

        self.tau = tau
        self._decay = np.exp(-dt / self.tau).astype( np.float32 )
        self.thr = thr

        #with tf.variable_scope('InputWeights'):
        self.w_in_var = torch.nn.Parameter(torch.randn(n_in, n_rec, dtype=dtype) / np.sqrt(n_in))
        self.w_in_val = torch.nn.Identity(self.w_in_var)

        #with tf.variable_scope('RecWeights'):
        self.w_rec_var = torch.nn.Parameter(torch.randn(n_rec, n_rec, dtype=dtype) / np.sqrt(n_rec))
        self.recurrent_disconnect_mask = torch.diag(torch.ones(n_rec)).bool()
        self.w_rec_mask = torch.ones(n_rec, n_rec) - torch.diag(torch.ones(n_rec))
        self.w_rec_val = torch.where(self.recurrent_disconnect_mask, torch.zeros_like(self.w_rec_var),
                                  self.w_rec_var)  # Disconnect autotapse

    def state_size(self):
        return LightLIFStateTuple(v=self.n_rec, z=self.n_rec)

    def output_size(self):
        return [self.n_rec, self.n_rec]

    def zero_state(self, batch_size, dtype, n_rec=None):
        if n_rec is None: n_rec = self.n_rec

        v0 = torch.zeros(size=(batch_size, n_rec), dtype=dtype)
        z0 = torch.zeros(size=(batch_size, n_rec), dtype=dtype)

        return LightLIFStateTuple(v=v0, z=z0)

    def forward(self, inputs, state, scope=None, dtype=torch.float32):
        # state in tensorflow comes from the RNN cell module. We don't have that in pytorch,
        # so that will be replaced by a simple tuple
        thr = self.thr
        z = state.z
        v = state.v
        decay = self._decay

        if self.stop_z_gradients:
            z = z.detach()

        # update the voltage
        i_t = torch.matmul(inputs, self.w_in_var) + torch.matmul(z, self.w_rec_var*self.w_rec_mask)
        I_reset = z * self.thr * self.dt
        new_v = decay * v + (1 - decay) * i_t - I_reset

        # Spike generation
        v_scaled = (new_v - thr) / thr
        new_z = SpikeFunction(v_scaled, self.thr)
        new_z = new_z * 1 / self.dt
        new_state = LightLIFStateTuple(v=new_v, z=new_z)
        return [new_z, new_v], new_state
    
    
    
LightALIFStateTuple = namedtuple('LightALIFState', ('z','v','b'))
class LightALIF(torch.nn.Module):
    def __init__(self, n_in, n_rec, tau=20., thr=0.03, dt=1., dtype=torch.float32, dampening_factor=0.3,
                 tau_adaptation=200., beta=1.6, stop_z_gradients=False):

        super(LightALIF, self).__init__()
        
        self.dampening_factor = dampening_factor
        self.dt = float(dt)
        self.n_in = n_in
        self.n_rec = n_rec
        self.data_type = dtype
        self.stop_z_gradients = stop_z_gradients

        self._num_units = self.n_rec

        self.tau = tau
        self._decay = np.exp(-dt / self.tau).astype( np.float32 )
        self.thr = thr
                
        self.tau_adaptation = tau_adaptation
        self.beta = beta
        self.decay_b = np.exp(-dt / tau_adaptation)

        #with tf.variable_scope('InputWeights'):
        self.w_in_var = torch.nn.Parameter(torch.randn(n_in, n_rec, dtype=dtype) / np.sqrt(n_in))
        self.w_in_val = torch.nn.Identity(self.w_in_var)

        #with tf.variable_scope('RecWeights'):
        self.w_rec_var = torch.nn.Parameter(torch.randn(n_rec, n_rec, dtype=dtype) / np.sqrt(n_rec))
        self.recurrent_disconnect_mask = torch.diag(torch.ones(n_rec)).bool()
        self.w_rec_val = torch.where(self.recurrent_disconnect_mask, torch.zeros_like(self.w_rec_var),
                                  self.w_rec_var)  # Disconnect autotapse

    def state_size(self):
        return LightALIFStateTuple(v=self.n_rec, z=self.n_rec, b=self.n_rec)

    def output_size(self):
        return [self.n_rec, self.n_rec, self.n_rec]

    def zero_state(self, batch_size, dtype):
        v0 = torch.zeros(size=(batch_size, self.n_rec), dtype=dtype)
        z0 = torch.zeros(size=(batch_size, self.n_rec), dtype=dtype)
        b0 = torch.zeros(size=(batch_size, self.n_rec), dtype=dtype)
        return LightALIFStateTuple(v=v0, z=z0, b=b0)


    def forward(self, inputs, state, scope=None, dtype=torch.float32):
        z = state.z
        v = state.v
        b = state.b
        decay = self._decay

        # the eligibility traces of b see the spike of the own neuron
        new_b = self.decay_b * b + (1. - self.decay_b) * z
        thr = self.thr + new_b * self.beta
        if self.stop_z_gradients:
            z = z.detach()

        # update the voltage
        i_t = torch.matmul(inputs, self.w_in_var) + torch.matmul(z, self.w_rec_var*self.recurrent_disconnect_mask)
        I_reset = z * self.thr * self.dt
        new_v = decay * v + (1 - decay) * i_t - I_reset

        # Spike generation
        v_scaled = (new_v - thr) / thr
        new_z = SpikeFunction(v_scaled, thr)
        new_z = new_z * 1 / self.dt

        new_state = LightALIFStateTuple(v=new_v,z=new_z, b=new_b) 
        return [new_z, new_v, new_b], new_state
    
    
    
EligALIFStateTuple = namedtuple('EligALIFStateTuple', ('s', 'z', 'z_local', 'r'))

class EligALIF(torch.nn.Module):
    def __init__(self, n_in, n_rec, tau=20., thr=0.03, dt=1., dtype=torch.float32, dampening_factor=0.3,
                 tau_adaptation=200., beta=1.6,
                 stop_z_gradients=False, n_refractory=1):
        super(EligALIF, self).__init__()

        if tau_adaptation is None: raise ValueError("alpha parameter for adaptive bias must be set")
        if beta is None: raise ValueError("beta parameter for adaptive bias must be set")
            
        if np.isscalar(tau): tau = torch.ones(n_rec, dtype=dtype) * np.mean(tau)
        if np.isscalar(thr): thr = torch.ones(n_rec, dtype=dtype) * np.mean(thr)

        #tau = torch.tensor( tau ).type(dtype=dtype)  #tau.type(dtype=dtype)
        #dt = torch.tensor( dt ).type(dtype=dtype) #dt = dt.type(dtype=dtype)

        self.n_refractory = n_refractory
        self.tau_adaptation = tau_adaptation
        self.beta = torch.tensor( beta ).type(dtype=dtype)
        self.decay_b = np.exp(-dt / tau_adaptation).astype(np.float32)

        self.dampening_factor = dampening_factor
        self.stop_z_gradients = stop_z_gradients
        self.dt = dt
        self.n_in = n_in
        self.n_rec = n_rec
        self.data_type = dtype

        self._num_units = self.n_rec

        self.tau = tau
        self._decay = np.exp(-dt / tau).astype(np.float32)
        self.thr = thr

        #with tf.variable_scope('InputWeights'):
        self.w_in_var = torch.nn.Parameter(torch.randn(n_in, n_rec, dtype=dtype) / np.sqrt(n_in))
        self.w_in_val = torch.nn.Identity(self.w_in_var)

        #with tf.variable_scope('RecWeights'):
        self.w_rec_var = torch.nn.Parameter(torch.randn(n_rec, n_rec, dtype=dtype) / np.sqrt(n_rec))
        self.recurrent_disconnect_mask = torch.diag(torch.ones(n_rec)).bool()
        self.w_rec_mask = torch.ones(n_rec) - torch.diag(torch.ones(n_rec))
        self.w_rec_val = torch.where(self.recurrent_disconnect_mask, torch.zeros_like(self.w_rec_var),
                                  self.w_rec_var)  # Disconnect self-connection

        self.variable_list = [self.w_in_var, self.w_rec_var]
        self.built = True

    def state_size(self):
        return EligALIFStateTuple(s=[self.n_rec, 2], 
                                  z=self.n_rec, r=self.n_rec, z_local=self.n_rec)

    def output_size(self):
        return [self.n_rec, [self.n_rec, 2]]

    def zero_state(self, batch_size, dtype, n_rec=None):
        if n_rec is None: n_rec = self.n_rec

        s0 = torch.zeros(size=(batch_size, n_rec, 2), dtype=dtype)
        z0 = torch.zeros(size=(batch_size, n_rec), dtype=dtype)
        z_local0 = torch.zeros(size=(batch_size, n_rec), dtype=dtype)
        r0 = torch.zeros(size=(batch_size, n_rec), dtype=dtype)

        return EligALIFStateTuple(s=s0, z=z0, r=r0, z_local=z_local0)

    def compute_z(self, v, b):
        adaptive_thr = self.thr + b * self.beta
        v_scaled = (v - adaptive_thr) / self.thr
        z = SpikeFunction(v_scaled, 0)
        z = z * 1 / self.dt
        return z

    def compute_v_relative_to_threshold_values(self,hidden_states):
        v = hidden_states[..., 0]
        b = hidden_states[..., 1]

        adaptive_thr = self.thr + b * self.beta
        v_scaled = (v - adaptive_thr) / self.thr
        return v_scaled

    def forward(self, inputs, state, scope=None, dtype=torch.float32, stop_gradient=None):

        decay = self._decay
        z = state.z
        z_local = state.z_local
        s = state.s
        r = state.r
        v, b = s[..., 0], s[..., 1]

        # This stop_gradient allows computing e-prop with auto-diff.
        #
        # needed for correct auto-diff computation of gradient for threshold adaptation
        # stop_gradient: forward pass unchanged, gradient is blocked in the backward pass
        use_stop_gradient = stop_gradient if stop_gradient is not None else self.stop_z_gradients
        if use_stop_gradient:
            z = z.detach()

        new_b = self.decay_b * b + z_local # threshold update does not have to depend on the stopped-gradient-z, it's local

        i_t = torch.matmul(inputs, self.w_in_var) + torch.matmul(z, self.w_rec_val*self.w_rec_mask) # gradients are blocked in spike transmission
        I_reset = z * self.thr * self.dt
        new_v = decay * v + i_t - I_reset

        # Spike generation
        is_refractory = (r > 0)
        zeros_like_spikes = torch.zeros_like(z)
        new_z = torch.where(is_refractory, zeros_like_spikes, self.compute_z(new_v, new_b))
        new_z_local = torch.where(is_refractory, zeros_like_spikes, self.compute_z(new_v, new_b))
        new_r = r + self.n_refractory * new_z - 1
        new_r = torch.clamp(new_r, 0., float(self.n_refractory))
        new_r = new_r.detach()
        new_s = torch.stack((new_v, new_b), dim=-1)

        new_state = EligALIFStateTuple(s=new_s, z=new_z, r=new_r, z_local=new_z_local)
        return [new_z, new_s], new_state

    
    def compute_eligibility_traces(self, v_scaled, z_pre, z_post, is_rec):

        n_neurons = z_post.size(2)
        rho = self.decay_b
        beta = self.beta
        alpha = self._decay
        n_ref = self.n_refractory

        # everything should be time major
        z_pre = z_pre.permute(1, 0, 2)
        v_scaled = v_scaled.permute(1, 0, 2)
        z_post = z_post.permute(1, 0, 2)

        psi_no_ref = self.dampening_factor / self.thr * torch.maximum(torch.tensor([0.]), 1. - torch.abs(v_scaled))
        
        update_refractory = lambda refractory_count, z_post:\
            torch.where(z_post > 0,torch.ones_like(refractory_count) * (n_ref - 1),torch.maximum(torch.tensor(0).int(), refractory_count - 1))

        refractory_count_init = torch.zeros_like(z_post[0], dtype=torch.int32)
        #refractory_count = tf.scan(update_refractory, z_post[:-1], initializer=refractory_count_init)
        #refractory_count_upd = update_refractory( refractory_count_init, z_post[:-1])
        refractory_count = []
        refractory_count_upd = update_refractory( refractory_count_init, z_post[0])
        refractory_count.append( refractory_count_upd )
        for i in range( z_post[:-1].size(0) ):
            refractory_count_upd = update_refractory( refractory_count_upd, z_post[i])
            refractory_count.append( refractory_count_upd )
        refractory_count = torch.stack( refractory_count, dim=0 )
        #refractory_count = torch.cat((refractory_count_init.unsqueeze(0), refractory_count_upd), dim=0)

        is_refractory = refractory_count > 0
        psi = torch.where(is_refractory, torch.zeros_like(psi_no_ref), psi_no_ref)

        update_epsilon_v = lambda epsilon_v, z_pre: alpha[None, None, :] * epsilon_v + z_pre[:, :, None]
        epsilon_v_zero = torch.ones((1, 1, n_neurons)) * z_pre[0][:, :, None]
        #epsilon_v = tf.scan(update_epsilon_v, z_pre[1:], initializer=epsilon_v_zero, )
        epsilon_v = []
        epsilon_v.append( epsilon_v_zero )
        epsilon_past = update_epsilon_v( epsilon_v_zero, z_pre[1] )
        epsilon_v.append( epsilon_past )
        for i in range( 2,z_pre.size(0) ):
            epsilon_past = update_epsilon_v( epsilon_past, z_pre[i] )
            epsilon_v.append( epsilon_past )
        epsilon_v = torch.stack( epsilon_v, dim=0 )

        update_epsilon_a = lambda epsilon_a, elems:\
                (rho - beta * elems['psi'][:, None, :]) * epsilon_a + elems['psi'][:, None, :] * elems['epsi']

        epsilon_a_zero = torch.zeros_like(epsilon_v[0])
        epsilon_a = []
        epsilon_a.append( epsilon_a_zero )
        epsilon_a_upd = update_epsilon_a( epsilon_a_zero, elems={ 'psi': psi[i], 'epsi': epsilon_v[i], 'previous_epsi':shift_by_one_time_step(epsilon_v[i]) } )
        epsilon_a.append( epsilon_a_upd )
        for i in range( 1,epsilon_v.size(0)-1 ):
            epsilon_a_upd = update_epsilon_a( epsilon_a_upd, elems={ 'psi': psi[i], 'epsi': epsilon_v[i], 'previous_epsi':shift_by_one_time_step(epsilon_v[i]) } )
            epsilon_a.append( epsilon_a_upd )
        epsilon_a = torch.stack( epsilon_a, dim=0 )

        e_trace = psi[:, :, None, :] * (epsilon_v - beta * epsilon_a)

        # everything should be time major
        e_trace = e_trace.permute( 1, 0, 2, 3 )
        epsilon_v = epsilon_v.permute( 1, 0, 2, 3 )
        epsilon_a = epsilon_a.permute( 1, 0, 2, 3 )
        psi = psi.permute( 1, 0, 2 )

        if is_rec:
            identity_diag = torch.eye(n_neurons)[None, None, :, :]
            e_trace -= identity_diag * e_trace
            epsilon_v -= identity_diag * epsilon_v
            epsilon_a -= identity_diag * epsilon_a

        return e_trace, epsilon_v, epsilon_a, psi
    
    
    def compute_eligibility_traces_pcm(self, v_scaled, z_pre, z_post, is_rec, nu=0.15, delta_eps=1):

        n_neurons = z_post.size(2)
        rho = self.decay_b
        beta = self.beta
        alpha = self._decay
        n_ref = self.n_refractory

        # everything should be time major
        z_pre = z_pre.permute(1, 0, 2)
        v_scaled = v_scaled.permute(1, 0, 2)
        z_post = z_post.permute(1, 0, 2)

        psi_no_ref = self.dampening_factor / self.thr * torch.maximum(torch.tensor([0.]), 1. - torch.abs(v_scaled))
        
        update_refractory = lambda refractory_count, z_post:\
            torch.where(z_post > 0,torch.ones_like(refractory_count) * (n_ref - 1),torch.maximum(torch.tensor(0).int(), refractory_count - 1))

        refractory_count_init = torch.zeros_like(z_post[0], dtype=torch.int32)
        #refractory_count = tf.scan(update_refractory, z_post[:-1], initializer=refractory_count_init)
        #refractory_count_upd = update_refractory( refractory_count_init, z_post[:-1])
        refractory_count = []
        refractory_count_upd = update_refractory( refractory_count_init, z_post[0])
        refractory_count.append( refractory_count_upd )
        for i in range( z_post[:-1].size(0) ):
            refractory_count_upd = update_refractory( refractory_count_upd, z_post[i])
            refractory_count.append( refractory_count_upd )
        refractory_count = torch.stack( refractory_count, dim=0 )
        #refractory_count = torch.cat((refractory_count_init.unsqueeze(0), refractory_count_upd), dim=0)

        is_refractory = refractory_count > 0
        psi = torch.where(is_refractory, torch.zeros_like(psi_no_ref), psi_no_ref)

        ### epsilon_v vectorized, to be developed ###
        ###update_epsilon_v = lambda epsilon_v, z_pre: alpha[None, None, :] * epsilon_v + z_pre[:, :, None]
        #epsilon_v = torch.ones(z_pre.size(0), z_pre.size(1), n_neurons, n_neurons).type(dtype=self.data_type)
        #Tevolve = torch.zeros( z_pre.size(1), n_neurons )
        #eps_ref = torch.zeros( z_pre.size(1), n_neurons )
        #for t in range( z_pre.size(0) ):
        #    b_spk, idx_spk  = torch.where( z_pre[t] == 1 )
        #    b_notspk, idx_notspk = torch.where( z_pre[t] != 1 )
        #    Tevolve[b_spk, idx_spk] = 0.0
        #    Tevolve[b_notspk, idx_notspk] = Tevolve[b_notspk, idx_notspk] + cell.dt
        #    eps_ref[b_spk, idx_spk] = epsilon_v[t-1, b_spk, 0, idx_spk] + z_pre[t, b_spk, idx_spk]
        #    epsilon_v[t] = eps_ref * ( self.dt/( self.dt + Tevolve ) )**nu
        
        ### epsilon_v non vectorized ###
        epsilon_v = torch.zeros( z_pre.size(0), z_pre.size(1), n_neurons, n_neurons ).type(dtype=self.data_type)
        eps_ref = torch.zeros( z_pre.size(1), n_neurons ).type(dtype=self.data_type)
        Tevolve = torch.zeros( z_pre.size(1), n_neurons ).type(dtype=self.data_type)
        for i in range( n_neurons ):
            for j in range( n_neurons ):
                Tevolve[:,i] = 0.0
                eps_ref[:,i] = 0.0
                for t in range( z_pre.size(0) ):
                    spk = torch.where( z_pre[t,:,i] == 1 )[0]
                    notspk = torch.where( z_pre[t,:,i] != 1 )[0]
                    Tevolve[spk, i] = 0.0
                    Tevolve[notspk, i] = Tevolve[notspk, i] + self.dt
                    eps_ref[spk, i] = torch.add( epsilon_v[t-1, spk, i, j], 1 )
                    epsilon_v[t,:,i,j] = eps_ref[:, i] * ( self.dt/( self.dt + Tevolve[:, i] ) )**nu
        

        update_epsilon_a = lambda epsilon_a, elems:\
                (rho - beta * elems['psi'][:, None, :]) * epsilon_a + elems['psi'][:, None, :] * elems['epsi']

        epsilon_a_zero = torch.zeros_like(epsilon_v[0])
        epsilon_a = []
        epsilon_a.append( epsilon_a_zero )
        epsilon_a_upd = update_epsilon_a( epsilon_a_zero, elems={ 'psi': psi[i], 'epsi': epsilon_v[i], 'previous_epsi':shift_by_one_time_step(epsilon_v[i]) } )
        epsilon_a.append( epsilon_a_upd )
        for i in range( 1,epsilon_v.size(0)-1 ):
            epsilon_a_upd = update_epsilon_a( epsilon_a_upd, elems={ 'psi': psi[i], 'epsi': epsilon_v[i], 'previous_epsi':shift_by_one_time_step(epsilon_v[i]) } )
            epsilon_a.append( epsilon_a_upd )
        epsilon_a = torch.stack( epsilon_a, dim=0 )

        e_trace = psi[:, :, None, :] * (epsilon_v - beta * epsilon_a)

        # everything should be time major
        e_trace = e_trace.permute( 1, 0, 2, 3 )
        epsilon_v = epsilon_v.permute( 1, 0, 2, 3 )
        epsilon_a = epsilon_a.permute( 1, 0, 2, 3 )
        psi = psi.permute( 1, 0, 2 )

        if is_rec:
            identity_diag = torch.eye(n_neurons)[None, None, :, :]
            e_trace -= identity_diag * e_trace
            epsilon_v -= identity_diag * epsilon_v
            epsilon_a -= identity_diag * epsilon_a

        return e_trace, epsilon_v, epsilon_a, psi

    def compute_loss_gradient(self, learning_signal, z_pre, z_post, v_post, b_post,
                              decay_out=None,zero_on_diagonal=None):
        thr_post = self.thr + self.beta * b_post
        v_scaled = (v_post - thr_post) / self.thr

        e_trace, epsilon_v, epsilon_a, _ = self.compute_eligibility_traces(v_scaled, z_pre, z_post, zero_on_diagonal)

        if decay_out is not None:
            e_trace_time_major = e_trace.permute(1, 0, 2, 3)
            filtered_e_zero = torch.zeros_like(e_trace_time_major[0])
            ####################################################################################
            filtering = lambda filtered_e, e: filtered_e * decay_out + e * (1 - decay_out)
            filtered_e = tf.scan(filtering, e_trace_time_major, initializer=filtered_e_zero)
            ####################################################################################
            filtered_e = filtered_e.permute( 1, 0, 2, 3 )
            e_trace = filtered_e

        gradient = torch.einsum('btj,btij->ij', learning_signal, e_trace)
        return gradient, e_trace, epsilon_v, epsilon_a
                          
                          
def exp_convolve(tensor, decay):
    '''
    Filters a tensor with an exponential filter.
    :param tensor: a tensor of shape (trial, time, neuron)
    :param decay: a decay constant of the form exp(-dt/tau) with tau the time constant
    :return: the filtered tensor of shape (trial, time, neuron)
    '''
    #with tf.name_scope('ExpConvolve'):
    #    assert tensor.dtype in [tf.float16, tf.float32, tf.float64]
    r_shp = range(len(tensor.size()))
    transpose_perm = [1, 0] + list(r_shp)[2:]

    tensor_time_major = tensor.permute(transpose_perm)
    initializer = torch.zeros_like(tensor_time_major[0])
    #filtered_tensor = tf.scan(lambda a, x: a * decay + (1 - decay) * x, tensor_time_major, initializer=initializer)
    #########################################################################
    exp_kernel = lambda a, x: a * decay + (1 - decay) * x
    filtered_tensor = []
    tensor_upd = exp_kernel( initializer, tensor_time_major[0] )
    filtered_tensor.append( tensor_upd )
    for i in range( 1, tensor_time_major.size(0) ):
        tensor_upd = exp_kernel( tensor_upd, tensor_time_major[i] )
        filtered_tensor.append( tensor_upd )
    filtered_tensor = torch.stack( filtered_tensor, dim=0 )
    #########################################################################
    filtered_tensor = filtered_tensor.permute(transpose_perm)

    return filtered_tensor

def pcm_convolve(tensor, nu=0.1, delta_g=1, dt=1, dtype=torch.float32):
    '''
    Given a train of pulses, an eligibility vector is produced, where at each pre-synaptic pulse,
    a PCM device undergoes a progressive set, to then drift following a power-law behavior.
    
    In a first implementation, the parameters of the power law are tuned to maximise performance of
    pcm-prop, later they will be replaced with data calibrated on experimental evidence.
    FM
    '''
    r_shp = range(len(tensor.size()))
    transpose_perm = [1, 0] + list(r_shp)[2:]
    tensor_time_major = tensor.permute(transpose_perm)
    
    ### pcm-trace, non vectorized ###
    filtered_tensor = torch.zeros( tensor.size(1), tensor.size(0), tensor.size(-1) ).type(dtype=dtype)
    eps_ref = torch.zeros( tensor.size(0), tensor.size(-1) ).type(dtype=dtype)
    Tevolve = torch.zeros( tensor.size(0), tensor.size(-1) ).type(dtype=dtype)
    for t in range( tensor.size(1) ):
        spk = torch.where( tensor_time_major[t] == 1 )
        notspk = torch.where( tensor_time_major[t] != 1 )
        Tevolve[spk] = 0.0
        Tevolve[notspk] = Tevolve[notspk] + dt
        eps_ref[spk] = torch.add( filtered_tensor[t-1][spk], delta_g )
        filtered_tensor[t] = eps_ref * torch.pow( dt/( dt + Tevolve ), nu )
    filtered_tensor = filtered_tensor.permute(transpose_perm)
    return filtered_tensor


def shift_by_one_time_step(tensor, initializer=None):
    '''
    Shift the input on the time dimension by one.
    :param tensor: a tensor of shape (trial, time, neuron)
    :param initializer: pre-prend this as the new first element on the time dimension
    :return: a shifted tensor of shape (trial, time, neuron)
    
    note: initializer has to be shaped as [trian, neuron], as it will be automatically
    unsqueezed inside the function
    '''
    #with tf.name_scope('TimeShift'):
    #    assert tensor.dtype in [tf.float16, tf.float32, tf.float64]
    # r_shp = range(len(tensor.size()))
    # transpose_perm = [1, 0] + list(r_shp)[2:]
    # tensor_time_major = tensor.permute(transpose_perm)
    shape = tensor.shape

    if initializer is None:
        # initializer = torch.zeros_like(tensor_time_major[0])
        initializer = torch.zeros(shape[1:])

    shifted_tensor = torch.cat([initializer.unsqueeze(0), tensor[:-1]], dim=0)

    # shifted_tensor = shifted_tensor.permute(transpose_perm)
    return shifted_tensor


def check_gradients(var_list, eprop_grads_np, true_grads_np):
    '''
    Check the correctness of the gradients.
    A ValueError() is raised if the gradients are not almost identical.

    :param var_list: the list of trainable tensorflow variables
    :param eprop_grads_np: a list of numpy arrays containing the gradients obtained eprop
    :param true_grads_np: a list of numpy arrays containing the gradients obtained with bptt
    :return: 
    '''
    for k_v, v in enumerate(var_list):
        eprop_grad = eprop_grads_np[k_v]
        true_grad = true_grads_np[k_v]

        diff = eprop_grad - true_grad
        is_correct = np.abs(diff) < 1e-4

        if np.all(is_correct):
            print('\t' + v.name + ' is correct.')
        else:
            print('\t' + v.name + ' is wrong')
            ratio = np.abs(eprop_grad) / (1e-8 + np.abs(true_grad))
            print('E-prop')
            print(np.array_str(eprop_grad[:5, :5], precision=4))
            print('True gradients')
            print(np.array_str(true_grad[:5, :5], precision=4))
            print('Difference')
            print(np.array_str(diff[:5, :5], precision=4))
            print('Ratio')
            print(np.array_str(ratio[:5, :5], precision=4))

            mismatch_indices = np.where(1 - is_correct)
            mismatch_indices = list(zip(*mismatch_indices))
            print('mismatch indices', mismatch_indices[:5])
            print('diff. vals', [diff[i, j] for i, j in mismatch_indices[:5]])

            raise ValueError()