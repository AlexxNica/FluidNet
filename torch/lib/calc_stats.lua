-- Copyright 2016 Google Inc, NYU.
-- 
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
-- 
--     http://www.apache.org/licenses/LICENSE-2.0
-- 
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- This is really just a 'catch all' function for a bunch of hacky data and
-- analysis (some of which was used for the paper).

local sys = require('sys')
local tfluids = require('tfluids')
local image = torch.loadPackageSafe('image')

-- Calculate divergence stats. 
function torch.calcStats(input)
  -- Unpack the input table.
  local data = input.data
  local conf = input.conf
  local mconf = input.mconf
  local model = input.model
  local nSteps = input.nSteps or 128

  torch.setDropoutTrain(model, false)

  local batchCPU = data:AllocateBatchMemory(1)
  local batchGPU = torch.deepClone(batchCPU, 'torch.CudaTensor')
  local divNet = tfluids.VelocityDivergence():cuda()

  -- The first thing we should do is evaluate a single sample that is a known
  -- failure case (lots of buoyancy).
  local isample = math.floor(data:nsamples() / 5)
  data:CreateBatch(batchCPU, torch.IntTensor({isample}), 1, conf.dataDir)
  torch.syncBatchToGPU(batchCPU, batchGPU)
  local p, U, flags = tfluids.getPUFlagsDensityReference(batchGPU)
  local density = p:clone():fill(1)  -- So that buoyancy adds a gradient.

  -- Now, do a step of PCG to remove any divergence.
  local div = p:clone()
  tfluids.velocityDivergenceForward(U, flags, div)
  local residual = tfluids.solveLinearSystemPCG(
      p, flags, div, mconf.is3D, 1e-6, 1000, 'ic0', false)
  tfluids.velocityUpdateForward(U, flags, p)

  tfluids.velocityDivergenceForward(U, flags, div)
  assert(div[1]:norm() < 1e-4, 'U field is not starting div free')

  -- Now add divergence with and without buoyancy.
  for withBuoyancy = 0, 1 do
    local curU = U:clone()
    local dt = 0.1
    tfluids.advectVel(dt, curU, flags, 'eulerOurs')
    if withBuoyancy == 1 then
      local gravity = torch.Tensor():typeAs(U):resize(3):fill(0)
      gravity[2] = -tfluids.getDx(flags)
      tfluids.addBuoyancy(curU, flags, density, gravity, dt)
    end

    tfluids.setWallBcsForward(curU, flags)

    -- Now use PCG again to get a ground truth output.
    tfluids.velocityDivergenceForward(curU, flags, div)
    local pPCG = p:clone()
    local residual = tfluids.solveLinearSystemPCG(
        pPCG, flags, div, mconf.is3D, 1e-6, 1000, 'ic0', false)
    local UPCG = curU:clone()
    tfluids.velocityUpdateForward(UPCG, flags, pPCG)
    tfluids.setWallBcsForward(UPCG, flags)

    -- Now use the Convnet to get the predicted output.
    local modelOutput = model:forward({p, curU, flags})
    local pConvNet, UConvNet = torch.parseModelOutput(modelOutput)
    -- We're free to set geometry cells to whatever we want.
    -- and lets subtract off the global mean (of fluid cells).
    local occupancy = p:clone()
    local invOccupancy = occupancy:clone():mul(-1):add(1)
    tfluids.flagsToOccupancy(flags, occupancy)
    pConvNet:cmul(invOccupancy)
    pPCG:cmul(invOccupancy)
    pConvNet:add(-(pConvNet:sum() / invOccupancy:sum()))
    pPCG:add(-(pPCG:sum() / invOccupancy:sum()))
    pConvNet:cmul(invOccupancy)
    pPCG:cmul(invOccupancy)

    -- Get the divergences.
    tfluids.velocityDivergenceForward(UPCG, flags, div)
    print('withBuoyancy = ' .. withBuoyancy)
    print('PCG norm(div(U)) = ' .. div[1]:norm())
    tfluids.velocityDivergenceForward(UConvNet, flags, div)
    print('PCG norm(div(U)) = ' .. div[1]:norm())

    -- Plot the outputs.
    if image ~= nil then
      local i = math.ceil(pPCG:size(3) / 2)
      image.display{image = pPCG[1][1][i], zoom = 10, legend = 'pPCG ' ..
                    'withBuoyancy  ' .. withBuoyancy}
      image.display{image = pConvNet[1][1][i], zoom = 10, legend = 'pConvNet' ..
                    'withBuoyancy  ' .. withBuoyancy}
    end
  end

  -- Now lets go through the dataset and get the statistics of output
  -- divergence, etc.
  local batchCPU = data:AllocateBatchMemory(conf.batchSize)
  local batchGPU = torch.deepClone(batchCPU, 'torch.CudaTensor')
  local dataInds
  if conf.maxSamplesPerEpoch < math.huge then
    dataInds = torch.range(1,
                           math.min(data:nsamples(), conf.maxSamplesPerEpoch))
  else
    dataInds = torch.range(1, data:nsamples())
  end
  -- For each sample we'll save a histogram of 10,000 bins of varying scales.
  local nHistBins = 10000
  local histData = {
    nHistBins = torch.DoubleTensor({nHistBins}),

    UTargetNormHist = torch.zeros(nHistBins):double(),
    UTargetNormMin = torch.DoubleTensor({0}),
    UTargetNormMax = torch.DoubleTensor({100}),

    pTargetHist = torch.zeros(nHistBins):double(),
    pTargetMin = torch.DoubleTensor({-10}),
    pTargetMax = torch.DoubleTensor({10}),

    pErrHist = torch.zeros(nHistBins):double(),
    pErrMin = torch.DoubleTensor({-10}),
    pErrMax = torch.DoubleTensor({10}),

    UErrNormHist = torch.zeros(nHistBins):double(),
    UErrNormMin = torch.DoubleTensor({0}),
    UErrNormMax = torch.DoubleTensor({100}),

    divHist = torch.zeros(nHistBins):double(),
    divMin = torch.DoubleTensor({-100}),
    divMax = torch.DoubleTensor({100}),
  }

  print('\n==> Calculating Stats: (gpu ' .. tostring(conf.gpu) .. ')')
  io.flush()
  local normDiv = torch.zeros(#dataInds, nSteps):double()
  local nBatches = 0

  for t = 1, #dataInds, conf.batchSize do
    if math.fmod(nBatches, 10) == 0 then
      collectgarbage()
    end
    torch.progress(t, #dataInds)

    local imgList = {}  -- list of images in the current batch
    for i = t, math.min(math.min(t + conf.batchSize - 1, data:nsamples()),
                        #dataInds) do
      table.insert(imgList, dataInds[i])
    end

    -- TODO(tompson): Parallelize CreateBatch calls (as in run_epoch.lua), which
    -- is not easy because the current parallel code doesn't guarantee batch
    -- order (might be OK, but it would be nice to know what err each sample
    -- has).
    data:CreateBatch(batchCPU, torch.IntTensor(imgList), conf.batchSize,
                     conf.dataDir)

    -- We want to save the error distribution.
    torch.syncBatchToGPU(batchCPU, batchGPU)
    local input = torch.getModelInput(batchGPU)
    local target = torch.getModelTarget(batchGPU)
    local output = model:forward(input)
    local pPred, UPred = torch.parseModelOutput(output)
    local pTarget, UTarget, flags = torch.parseModelTarget(target)
    local pErr = pPred - pTarget
    local UErr = UPred - UTarget
    local div = divNet:forward({UPred, flags}) 
     
    -- Record UTarget, pTarget, pErr, UErr, div.
    -- We can't store ALL the dataset voxels in memory. So we need to create
    -- some sort of compressed representation of them that's still helpful
    -- statistics for debugging.
    -- --> The best I could come up with is a histogram.
    local function createHist(data, min, max)
      if data:size(1) > 1 then
        data = torch.norm(data, 1, 2)  -- Record histogram of L2 mag.
      end
      data = data:view(data:numel())  -- Vectorize to 1D.
      local hist = torch.histc(data:float(), nHistBins, min[1], max[1]):double()
      return hist
    end
    for i = 1, #imgList do
      local hist = createHist(UTarget[i], histData.UTargetNormMin,
                              histData.UTargetNormMax)
      histData.UTargetNormHist:add(hist)

      hist = createHist(pTarget[i], histData.pTargetMin,
                        histData.pTargetMax)
      histData.pTargetHist:add(hist)          

      hist = createHist(pErr[i], histData.pErrMin, histData.pErrMax)
      histData.pErrHist:add(hist)    

      hist = createHist(UErr[i], histData.UErrNormMin, histData.UErrNormMax)
      histData.UErrNormHist:add(hist)

      hist = createHist(div[i], histData.divMin, histData.divMax)
      histData.divHist:add(hist)    
    end

    -- Now record divergence stability vs time.
    -- Restart the sim from the target frame.
    local p, U, flags, density = tfluids.getPUFlagsDensityReference(batchGPU)
    U:copy(batchGPU.UTarget)
    p:copy(batchGPU.pTarget)

    -- Record the divergence of the start frame.
    for i = 1, #imgList do
      normDiv[{imgList[i], 1}] = div[i]:norm()
    end

    for j = 2, nSteps do
      local outputDiv = false
      tfluids.simulate(conf, mconf, batchGPU, model, outputDiv)
      local p, U, flags, density =
          tfluids.getPUFlagsDensityReference(batchGPU)
      div = divNet:forward({U, flags})
      for i = 1, #imgList do
        normDiv[{imgList[i], j}] = div[i]:norm()
      end
    end
    nBatches = nBatches + 1
  end
  torch.progress(#dataInds, #dataInds)  -- Finish the progress bar.

  return {normDiv = normDiv, histData = histData}
end
