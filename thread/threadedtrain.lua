local Threads = require 'threads'

local function threadedTrain(module, criterion, parameters)
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

            function pass(data)
                local output = module:forward(data)
                local gap = criterion:forward(output, parameters["label"])
                module:zeroGradParameters()
                module:backward(data, criterion:backward(output, parameters["label"]))
                weights[1][data[1][1]]:add(-parameters["learning_rate"], dweights[1][data[1][1]])
                for i = 1, parameters["neg_sample_number"] do
                    weights[2][data[2][i]]:add(-parameters["learning_rate"], dweights[2][data[2][i]])
                end
                return gap
            end
        end

    )

    local function Split(line, sep)
        local t = {}; local i = 1
        for word in string.gmatch(line, "%a+") do
            t[i] = word; i = i + 1
        end
        return t
    end

    local function GenerateContexts(pos_context_id)
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

    local start = sys.clock()
    local weights = module:parameters()
    local file = io.open(parameters["corpus"], "r")
    for iter = 1, parameters["epochs"] do
        local c = 0
        local t_err = 0
        for line in file:lines() do
            local sentence = Split(line)
            for i, word in ipairs(sentence) do
                local word_idx = parameters["word2index"][word]
                if word_idx ~= nil then
                    local center_word = torch.IntTensor(1)
                    center_word[1] = word_idx
                    local cur_window_size = torch.random(parameters["window_size"])
                    for j = i - cur_window_size, i + cur_window_size do
                        local context = sentence[j]
                        if context ~= nil and j ~= i then
                            local context_id = parameters["word2index"][context]
                            if context_id ~= nil then
                                local contexts = GenerateContexts(context_id)
                                threads:addjob(
                                    function()
                                        return pass({center_word, contexts})
                                    end,
                                    function(gap)--, dweights, data)
                                        t_err = t_err + gap
                                        c = c + 1
                                        if (c % 100000 == 0) then
                                            print(string.format("Epoch %d: Error Rate %f", iter, t_err / c))
                                            print(sys.clock() - start)
                                        end
                                    end
                                )
                            end
                        end
                    end
                end
            end
        end
        threads:synchronize()
        print(string.format("TrainStream Cost %d", sys.clock() - start))
    end

    threads:terminate()
end

return threadedTrain

