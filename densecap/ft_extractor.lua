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
local json = require 'json'

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

local function pack_outputs( outputs )
  -- body
  --1 `   objectness_scores, 256x1
  --2   pos_roi_boxes, 
  --3   final_box_trans, 
  --4 `  final_boxes,  [~128]x4
  --5   lm_output,     xxx
  --6 `  gt_boxes,     [~128]x4
  --7 `  gt_labels,    [~128]x15
  --8 `  pos_roi_feats [~128]x512x7x7
  --9 `  pos_roi_codes [~128]x4096
  --10 ` global_feat   512x30x45
  local pack = {}
  pack['box_scores'] = outputs[1]:type('torch.FloatTensor'):totable() --to array
  pack['boxes_pred'] = outputs[4]:type('torch.FloatTensor'):totable()
  pack['boxes_gt'] = outputs[6]:type('torch.FloatTensor'):totable()
  pack['box_captions_gt'] = outputs[7]:type('torch.FloatTensor'):totable()
  pack['box_feats'] = outputs[8]:type('torch.FloatTensor'):totable()
  pack['box_codes'] = outputs[9]:type('torch.FloatTensor'):totable()
  pack['glob_feat'] = outputs[10]:type('torch.FloatTensor'):totable()
  -- pack['glob_caption_gt'] = outputs[11]:type('torch.FloatTensor'):totable()

  return pack
end

local function check_pick_confirm(idx)
  -- body
  local fr = io.open('pick_confirm_'..tostring(idx), "r")
   if fr~=nil then io.close(f) return true else return false end
end

local function saveJson(outputs, idx)
  -- body
  -- check "confirm" file
  while true do
    if check_pick_confirm(idx) then
      os.remove('pick_confirm_'..tostring(idx))
      break
    end
  end
  
  local enc = json.encode(outputs)
  local f = io.open('data_'..tostring(idx),"w")
  f:write(enc)
  f:close()

end
-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local pipeLen = 10
local counter = 0
while true do  
  print(counter)
  local outputs = extractFeats()
  -- print('success!')
  out_packed = pack_outputs(outputs)
  -- save to json file
  saveJson(outputs, counter%pipeLen)
  counter = counter + 1

end
