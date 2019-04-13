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

require 'densecap.DataLoader'
require 'densecap.DenseCapModel'
require 'densecap.optim_updates'

local json = require 'cjson'
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
  local info = {}
  local data = {}
  data.image, data.gt_boxes, data.gt_labels, info.im_info, data.region_proposals, info.fullcap = loader:getBatch()
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

  return outputs_all, info --losses, stats
end

local function pack_outputs( outputs, info )
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
  pack['info'] = info['im_info'][1]
  print(info['im_info'])
  pack['box_scores'] = outputs[1]--:totable() --to array
  pack['boxes_pred'] = outputs[4]--:totable()
  pack['boxes_gt'] = outputs[6]--:totable()
  pack['box_captions_gt'] = outputs[7]--:totable()
  pack['box_feats'] = outputs[8]--:totable()
  pack['box_codes'] = outputs[9]--:totable()
  pack['glob_feat'] = outputs[10]--:totable() --:type('torch.FloatTensor')
  if info['fullcap']~= nil then
    pack['glob_caption_gt'] = info['fullcap']--:totable()
  end
  return pack
end

local function check_pick_confirm(idx)
  -- body
  local fr = io.open('./data_pipeline/pick_confirm_'..tostring(idx), "r")
  if fr~=nil then 
    os.remove('./data_pipeline/pick_confirm_'..tostring(idx))
    io.close(fr) 
    return true 
  else 
    return false 
  end
end

local function saveJson(outputs, pipeLen, odd)
  -- body
  -- check "confirm" file
  local idx = 0
  local inc = 1
  if odd>0 then
    inc = 4
    idx = odd
  end
  -- if odd>0 then idx = odd end
  while true do
    if check_pick_confirm(idx) then
      break
    end
    idx = (idx+inc)%pipeLen
    if odd==0 and idx%2==0 then
      idx = (idx+2)%pipeLen
    end
  end
  
  -- local enc = json.encode(outputs)
  local iter = outputs['info']['split_bounds'][1]
  local true_imid = outputs['info']['split_bounds'][2]
  local numiters = outputs['info']['split_bounds'][3]
  local fblk = io.open('./data_pipeline/writing_block_'..tostring(idx),"w")
  fblk:close()
  torch.save('./data_pipeline/data_'..tostring(idx)..'_'..tostring(true_imid)..'_'..tostring(numiters), outputs)
  -- local f = io.open('./data_pipeline/data_'..tostring(idx)..'_'..tostring(iter)..'_'..tostring(numiters),"w")
  -- f:write(enc)
  -- f:close()
  os.remove('./data_pipeline/writing_block_'..tostring(idx))

end
-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local pipeLen = opt.pipe_len
local counter = 0
while true do  
  print(counter)
  local outputs, info = extractFeats()
  -- print('success!')
  out_packed = pack_outputs(outputs, info)
  -- save to json file
  saveJson(out_packed, pipeLen, opt.odd)
  counter = counter + 1

end
