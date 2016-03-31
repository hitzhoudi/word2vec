local Threads = require 'threads'

local function split(line, sep)
    local t = {}; local i = 1
    for word in string.gmatch(line, "%a+") do
        t[i] = word; i = i + 1
    end
    return t
end

local function generateContexts(pos_context_id)
    local contexts = torch.IntTensor(1 + parameters["neg_sample_number"])
    contexts[1] = pos_context_id
    local i = 0
    while i < parameters["neg_sample_number"] do
        local neg_context_id = parameters["table"][torch.random(parameters["table_size"])]
        if neg_context_id ~= pos_context_id then
            contexts[i + 2] = neg_context_id
            i = i + 1
        end
    end
    return contexts
end

local function loadData(parameters)
    local start = sys.clock()
    local c = 0
    local file = io.open(parameters["corpus"], "r")
    local center_word = torch.IntTensor(1)

    local data = {}

    for line in file:lines() do
        local sentence = split(line)
        for i, word in ipairs(sentence) do
            local word_idx = parameters["word2index"][word]
            if word_idx ~= nil then
                center_word[1] = word_idx
                local cur_window_size = torch.random(parameters["window_size"])
                for j = i - cur_window_size, i + cur_window_size do
                    local context = sentence[j]
                    if context ~= nil and j ~= i then
                        local context_id = parameters["word2index"][context]
                        if context_id ~= nil then
                            c = c + 1
                            data[c] = {center_word, generateContexts(context_id)}
                        end
                        if c % 100000 == 0 then
                            print(string.format("Cost %d", sys.clock() - start))
                        end
                    end
                end
            end
        end
    end
    file:close()
    return data
end

local function threadedTrainUnit(module, criterion, parameters, data)
    Threads.serialization('threads.sharedserialize')
    local threads = Threads(
        parameters["thread_num"],
        function()
            require 'nn'
        end,

        function()
            local module = module:clone('weight', 'bais')
            local weights, dweights = module:parameters()
            local criterion = criterion:clone()
            local label = parameters["label"]

            function pass(idx)
--[[                local output = module:forward(data)
                local gap = criterion:forward(output, parameters["label"])
                module:zeroGradParameters()
                module:backward(data, criterion:backward(output, parameters["label"]))
                return gap, dweights ]]--
            end
        end
    )

    local start = sys.clock()
    local weights = module:parameters()
    local idx = 1
    while idx < #data / 5 do
         threads:addjob(
            function(idx)
                return pass(idx)
            end--,
                --function(gap, dweights)
                --[[function()
                    t_err = t_err + gap
                    for j = 1, #weights do
                        weights[j]:add(parameters["learning_rate"], dweights[j])
                    end
                    c = c + 1
                    if (c % 100000 == 0) then
                        print(sys.clock() - start)
                    end
                end]]--
            )
        idx = idx + 1
    end

    threads:synchronize()

--[[
                               )
        print(string.format("TrainStream Cost %d", sys.clock() - start)) ]]--
end

local function threadedTrain(module, criterion, parameters)
    for iter = 1, parameters["epochs"] do
        local data = loadData(parameters)
        threadedTrainUnit(module, criterion, parameters, data)
    end
    threads:terminate()
end

return threadedTrain

