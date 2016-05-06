rbm = {}
rbm.__index = rbm

require 'torch'
require 'cutorch'
require 'randomkit'

setmetatable(rbm, {
  __call = function (cls, ...)
    return cls.new(...)
  end,
})

function rbm.new(n_visible, n_hidden)
  local self = setmetatable({}, rbm)
  self.W = torch.Tensor(n_visible, n_hidden):zeros()
  self.hbias = torch.Tensor(n_hidden):zeros()
  self.vbias = torch.Tensor(n_visible):zeros()
  return self
end

-- Function to compute the free energy
function rbm:free_energy(v_sample)
  local wx_b = torch.mm(v_sample, self.W) + self.hbias
  local vbias_term = torch.mm(v_sample, self.vbias)
  local hidden_term = torch.sum(torch.log(1 + torch.exp(wx_b)))
  return -hidden_term - vbias_term
end

-- This function propagates visible units activation upwards to hidden units
function rbm:propup(vis)
  local pre_sigmoid_activation = torch.mm(vis, self.W) + self.hbias
  return {pre_sigmoid_activation, torch.sigmoid(pre_sigmoid_activation)}
end

-- This function infers state of hidden units given visible units
function rbm:sample_h_given_v(v0_sample)
  local pre_sigmoid_h1, h1_mean = self:propup(v0_sample)
  local h1_sample = randomkit.binomial(torch.Tensor(h1_mean:size()), 1, h1_mean)
  return {pre_sigmoid_h1, h1_mean, h1_sample}
end

-- This function propagates the hidden units activation downwards to the visible units
function rbm:propdown(hid)
  local pre_sigmoid_activation = torch.mm(vis, self.W:transpose(1, 2)) + self.hbias
  return {pre_sigmoid_activation, torch.sigmoid(pre_sigmoid_activation)}
end

-- This function infers state of visible units given hidden units
function rbm:sample_v_given_h(h0_sample)
  local pre_sigmoid_v1, v1_mean = self:propdown(h0_sample)
  local v1_sample = randomkit.binomial(torch.Tensor(v1_mean:size()), 1, v1_mean)
  return {pre_sigmoid_v1, v1_mean, v1_sample}
end

-- This function implements one step of Gibbs sampling, starting from the hidden state
function rbm:gibbs_hvh(h0_sample)
  local pre_sigmoid_v1, v1_mean, v1_sample = self:sample_v_given_h(h0_sample)
  local pre_sigmoid_h1, h1_mean, h1_sample = self:sample_h_given_v(v1_sample)
  return {pre_sigmoid_v1, v1_mean, v1_sample, pre_sigmoid_h1, h1_mean, h1_sample}
end

-- This function implements one step of Gibbs sampling, starting from the visible state
function rbm:gibbs_vhv(v0_sample)
  local pre_sigmoid_h1, h1_mean, h1_sample = self:sample_h_given_v(v0_sample)
  local pre_sigmoid_v1, v1_mean, v1_sample = self:sample_v_given_h(h1_sample)
  return {pre_sigmoid_h1, h1_mean, h1_sample, pre_sigmoid_v1, v1_mean, v1_sample}
end

-- This functions implements one step of CD-k or PCD-k
function rbm:get_cost_updates(lr, persistent, k)
  -- compute positive phase
  local pre_sigmoid_ph, ph_mean, ph_sample = self:sample_h_given_v(self.input)

  -- decide how to initialize persistent chain:
  -- for CD, we use the newly generate hidden sample
  -- for PCD, we initialize from the old state of the chain
  local chain_start = persistent

  if persistent then
    chain_start = ph_sample
  end

end
