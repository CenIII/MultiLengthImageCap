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

require 'hdf5'

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
local dtype = 'torch.FloatTensor'
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

local function loadImage(image_path)
  local vgg_mean = torch.FloatTensor{103.939, 116.779, 123.68} -- BGR order
  vgg_mean = vgg_mean:view(1,3,1,1)

  local h5_file = hdf5.open(image_path, 'r')
  -- local img = npy4th.loadnpy(image_path)--self.h5_file:read('/images'):partial({ix,ix},{1,self.num_channels},
                            --{1,self.max_image_size},{1,self.max_image_size})
  local img = h5_file:read('/image'):all()--:partial({1,1},{1,3},
                            --{1,self.max_image_size},{1,self.max_image_size})

  -- crop image to its original width/height, get rid of padding, and dummy first dim
  -- img = img[{ 1, {}, {1,self.image_heights[ix]}, {1,self.image_widths[ix]} }]
  img = img:float() -- convert to float
  -- img = img:view(1, img:size(1), img:size(2), img:size(3)) -- batch the image
  img:add(-1, vgg_mean:expandAs(img)) -- subtract vgg mean
  return img
end


local function extractFeats(image_path)
  model:evaluate()

  -- Fetch data using the loader
  local timer = torch.Timer()
  -- local info = {}
  local data = {}
  -- data.image, data.gt_boxes, data.gt_labels, info.im_info, data.region_proposals, info.fullcap = loader:getBatch()
  data.image = loadImage(image_path)
  for k, v in pairs(data) do
    data[k] = v:type(dtype)
  end
  if opt.timing then cutorch.synchronize() end
  local getBatch_time = timer:time().real

  -- Run the model forward and backward
  model.timing = opt.timing
  model.cnn_backward = false
  -- local losses, stats = model:forward_backward(data)
  local outputs_all = model:forward_demo(data)

  return outputs_all--, info --losses, stats
end

local function pack_outputs( outputs )
  --1 `   objectness_scores, 256x1
  --2 `  final_boxes,  [~128]x4
  --3 `  pos_roi_feats [~128]x512x7x7
  --4 `  gt_boxes,     [~128]x4
  --5 `  gt_labels,    [~128]x15
  --6 ` global_feat   512x30x45
  
  local pack = {}
  pack['box_scores'] = outputs[1]
  pack['boxes_pred'] = outputs[2]
  pack['box_feats'] = outputs[3]
  pack['glob_feat'] = outputs[6]
  return pack
end
local function split(inputstr, sep)
  if sep == nil then
          sep = "%s"
  end
  local t={}
  for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
          table.insert(t, str)
  end
  return t
end
local function saveJson(outputs, image_path)
  local sptb = split(image_path,'/')
  local imname = split(sptb[#sptb],'.')[1]
  print(imname)
  local fblk = io.open('./data_pipeline/writing_block_'..imname,"w")
  fblk:close()
  torch.save('./data_pipeline/data_demo_'..imname, outputs)
  os.remove('./data_pipeline/writing_block_'..imname)

end
-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------

-- while true do  
local outputs = extractFeats(opt.image_path)
local out_packed = pack_outputs(outputs)
-- save to json file
saveJson(out_packed,opt.image_path)

-- end
