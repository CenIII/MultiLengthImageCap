require 'densecap.modules.PosSlicer'
require 'nn'

local M = {}

function M._buildRecognitionNet(nets)
  
  local roi_feats = nn.Identity()()
  local roi_boxes = nn.Identity()()
  local gt_boxes = nn.Identity()()
  local gt_labels = nn.Identity()()

  local roi_codes = nets.recog_base(roi_feats)
  local objectness_scores = nets.objectness_branch(roi_codes)

  -- local pos_roi_feats = nn.PosSlicer(){roi_feats, gt_labels}

  local pos_roi_codes = nn.PosSlicer(){roi_codes, gt_labels}
  local pos_roi_boxes = nn.PosSlicer(){roi_boxes, gt_boxes}
  
  local final_box_trans = nets.box_reg_branch(pos_roi_codes)
  local final_boxes = nn.ApplyBoxTransform(){pos_roi_boxes, final_box_trans}

  -- local lm_input = {pos_roi_codes, gt_labels}
  -- local lm_output = 0 --self.nets.language_model(lm_input)

  -- Annotate nodes
  roi_codes:annotate{name='recog_base'}
  objectness_scores:annotate{name='objectness_branch'}
  pos_roi_codes:annotate{name='code_slicer'}
  pos_roi_boxes:annotate{name='box_slicer'}
  final_box_trans:annotate{name='box_reg_branch'}

  local inputs = {roi_feats, roi_boxes, gt_boxes, gt_labels}
  local outputs = {
    objectness_scores,
    pos_roi_boxes, final_box_trans, final_boxes,
    roi_codes,
    gt_boxes, gt_labels
    -- pos_roi_feats,
    -- pos_roi_codes
  }
  local mod = nn.gModule(inputs, outputs)
  mod.name = 'recognition_network'
  return mod
  -- body
end
function M.setup(opt)
  local model
  if opt.checkpoint_start_from == '' then
    print('initializing a DenseCap model from scratch...')
    model = DenseCapModel(opt)
  else
    print('initializing a DenseCap model from ' .. opt.checkpoint_start_from)
    -- new_model = DenseCapModel(opt)
    model = torch.load(opt.checkpoint_start_from).model
    model.opt.end_objectness_weight = opt.end_objectness_weight
    model.nets.localization_layer.opt.mid_objectness_weight = opt.mid_objectness_weight
    model.nets.localization_layer.opt.mid_box_reg_weight = opt.mid_box_reg_weight
    model.crits.box_reg_crit.w = opt.end_box_reg_weight
    local rpn = model.nets.localization_layer.nets.rpn
    rpn:findModules('nn.RegularizeLayer')[1].w = opt.box_reg_decay
    model.opt.train_remove_outbounds_boxes = opt.train_remove_outbounds_boxes
    model.opt.captioning_weight = opt.captioning_weight
    model.nets.posslice_net = nn.PosSlicer()
    tmpnet = nn.Sequential()
    tmpnet:add(model.nets.conv_net1)
    tmpnet:add(model.nets.conv_net2)
    tmpnet:add(model.nets.localization_layer)
    tmpnet:add(M._buildRecognitionNet(model.nets))
    model.net = tmpnet

    if cudnn then
      cudnn.convert(model.net, cudnn)
      cudnn.convert(model.nets.localization_layer.nets.rpn, cudnn)
    end
  end

  -- Find all Dropout layers and set their probabilities
  local dropout_modules = model.nets.recog_base:findModules('nn.Dropout')
  for i, dropout_module in ipairs(dropout_modules) do
    dropout_module.p = opt.drop_prob
  end
  model:float()

  return model
end

return M
