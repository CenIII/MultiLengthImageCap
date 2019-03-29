--[[
Main entry point for training a DenseCap model
]]--

-------------------------------------------------------------------------------
-- Includes
-------------------------------------------------------------------------------
require 'torch'
require 'nngraph'
require 'optim'
require 'image'
require 'lfs'
require 'nn'
local cjson = require 'cjson'

require 'densecap.DataLoader'
require 'densecap.DenseCapModel'
require 'densecap.optim_updates'
local utils = require 'densecap.utils'
local opts = require 'train_opts'
local models = require 'models'
local eval_utils = require 'eval.eval_utils'

require("mobdebug").start()

-------------------------------------------------------------------------------
-- Initializations
-------------------------------------------------------------------------------
local opt = opts.parse(arg)
print(opt)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
if opt.gpu >= 0 then
  -- cuda related includes and settings
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpu + 1) -- note +1 because lua is 1-indexed
end

-- initialize the data loader class
local loader = DataLoader(opt)
opt.seq_length = loader:getSeqLength()
opt.vocab_size = loader:getVocabSize()
opt.idx_to_token = loader.info.idx_to_token

-- initialize the DenseCap model object
local dtype = 'torch.CudaTensor'
local model = models.setup(opt):type(dtype)

-- get the parameters vector
local params, grad_params, cnn_params, cnn_grad_params = model:getParameters()
print('total number of parameters in net: ', grad_params:nElement())
print('total number of parameters in CNN: ', cnn_grad_params:nElement())

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local loss_history = {}
local all_losses = {}
local results_history = {}
local iter = 0
local function extractFeats()
  model:training()

  -- Fetch data using the loader
  local timer = torch.Timer()
  local info
  local data = {}
  data.image, data.gt_boxes, data.gt_labels, info, data.region_proposals = loader:getBatch()
  for k, v in pairs(data) do
    data[k] = v:type(dtype)
  end
  if opt.timing then cutorch.synchronize() end
  local getBatch_time = timer:time().real

  -- Run the model forward and backward
  model.timing = opt.timing
  model.cnn_backward = false
  -- local losses, stats = model:forward_backward(data)
  local outputs_all = model:forward_backward(data)

  return outputs_all --losses, stats
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
while true do  

  local outputs = extractFeats()
  
  print('success!')

end
