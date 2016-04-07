require("nn")
local Threads = require 'threads'

SkipGram = {}

local function Split(line, sep)
    local t = {}; local i = 1
    for word in string.gmatch(line, "%a+") do
        t[i] = word; i = i + 1
    end
    return t
end

local function BuildVocabulary(corpus_path, min_frequency)
    local start = sys.clock()
    local file = io.open(corpus_path, "r")
    local vocab = {}
    local total_count = 0
    for line in file:lines() do
        for _, word in ipairs(Split(line)) do
            total_count = total_count + 1
            if vocab[word] == nil then
                vocab[word] = 1
            else
                vocab[word] = vocab[word] + 1
            end
        end
    end
    file.close()
    local word2index = {}
    local index2word = {}
    for word, count in pairs(vocab) do
        if count >= min_frequency then
            index2word[#index2word + 1] = word;
            word2index[word] = #index2word;
        else
            vocab[word] = nil
        end
    end
    local vocab_size = #index2word
    print(string.format("BuildVacabulary costs: %d second(s)", sys.clock() - start))
    return vocab, vocab_size, word2index, index2word, total_count
end

local function BuildTable(vocab, word2index, unigram_model_power, table_size)
    local start = sys.clock()
    local total_count_power = 0
    for _, count in pairs(vocab) do
        total_count_power = total_count_power + count^unigram_model_power
    end

    local table = torch.IntTensor(table_size)
    local accum_word_prob = 0; accum_word_index = 0
    for word, count in pairs(vocab) do
        accum_word_prob = accum_word_prob + count^unigram_model_power / total_count_power
        while (accum_word_index + 1) / table_size < accum_word_prob do
            accum_word_index = accum_word_index + 1
            table[accum_word_index] = word2index[word]
        end
    end
    print(string.format("BuildTable costs: %d second(s)", sys.clock() - start))
    print(string.format("table size: %d", accum_word_index))
    return table, accum_word_index
end

local function ConstructModel(vocab_size, dim)
    local skip_gram = nn.Sequential()
    skip_gram:add(nn.ParallelTable())
    local word_vector = nn.LookupTable(vocab_size, dim)
    word_vector:reset(0.25);
    local context_vector = nn.LookupTable(vocab_size, dim)
    context_vector:reset(0.25)
    skip_gram.modules[1]:add(word_vector)
    skip_gram.modules[1]:add(context_vector)
    skip_gram:add(nn.MM(false, true))
    skip_gram:add(nn.Sigmoid())
    local criterion = nn.BCECriterion()

    return skip_gram, criterion
end

local function ReadData(corpus_path)
    local start = sys.clock()
    local file = io.open(corpus_path, "r")
    local data = {}
    local total_count = 0
    for line in file:lines() do
        for _, word in ipairs(Split(line)) do
            data[#data + 1] = word
        end
    end
    file.close()
    return data
end

local function ThreadedTrain(module, criterion, vocabulary, word2index, table, table_size, total_count, data, parameters)
    Threads.serialization('threads.sharedserialize')

    local function GenerateContexts(wid, cid)
        local contexts = torch.IntTensor(1 + parameters["neg_sample_number"])
        contexts[1] = cid
        local i = 0
        while i < parameters["neg_sample_number"] do
            local nid = table[torch.random(table_size)]
            if nid ~= cid and nid ~= wid then
                contexts[i + 2] = nid
                i = i + 1
            end
        end
        return contexts
    end

    local threads = Threads(
        parameters["thread_num"],
        function()
            require 'nn'
        end,
        function()
            local module = module:clone('weight', 'bais')
            local weights, dweights = module:parameters()
            local criterion = criterion:clone()
            local label = torch.zeros(1 + parameters["neg_sample_number"])
            label[1] = 1

            function pass()
                local s = #data / parameters["thread_num"] * (__threadid - 1) + 1
                local t = #data / parameters["thread_num"] * __threadid + 1
                t = math.min(t, #data + 1)
                local total_count = 0; count = 0; iter = 0
                while true do
                    -- update lr
                    if total_count > 0 and total_count % 10000 == 0 then
                        SkipGram.lr = parameters["learning_rate"] 
                    end

                    -- update iter
                    if s + count >= t then
                        count = 0
                        iter = iter + 1
                        if iter >= parameters["epochs"] then
                            break
                        end
                    end

                    -- extract 1000 words
                    local buffer = {}
                    while s + count < t and #buffer < 1000 do
                        word = data[s + count]
                        if vocabulary[word] ~= nil then
                            local ran = (1 + math.sqrt(vocabulary[word] / (parameters["sample"] * total_count)))
                                * parameters["sample"] * total_count / vocabulary[word]
                            --print(string.format("%s\t%f", word, ran))
                            if ran > math.random() then 
                                buffer[#buffer + 1] = word
                            end
                        end 
                        count = count + 1
                        total_count = total_count + 1
                    end
                   
                    -- train core
                    for i, word in ipairs(buffer) do
                        local wid = word2index[word]
                        local center_word = torch.IntTensor(1)
                        local cur_window_size = torch.random(parameters["window_size"])
                        for j = i - cur_window_size, i + cur_window_size do
                            if j >= 1 and j <= 1000 and j ~= i then
                                local cid = word2index[buffer[j]]
                                local contexts = GenerateContexts(wid, cid)
                                local output = module:forward({center_word, contexts})
                                local gap = criterion:forward(output, label)
                                module:zeroGradParameters()
                                module:backward(data, criterion:backward(output, label))
                                weights[1][data[1][1]]:add(-SkipGram.lr, dweights[1][data[1][1]])
                                for i = 1, parameters["neg_sample_number"] do
                                    weights[2][data[2][i]]:add(-SkipGram.lr, dweights[2][data[2][i]])
                                end
                            end
                        end
                    end
                end
            end
        end
    )

    local start = sys.clock()
    local weights = module:parameters()
    SkipGram.lr = parameters["learning_rate"]
    for tid = 1, parameters["thread_num"] do
        threads:addjob(
            function()
                return pass()
            end
        )
    end
    threads:synchronize()
    threads:terminate()
end

function SkipGram.Train()
    local vocab, vocab_size, word2index, index2word, total_count
        = BuildVocabulary(parameters["corpus"], parameters["min_frequency"])
    print("Total Count:" .. total_count .. " Vocabulary Size:" .. vocab_size)
    os.execute("mkdir " .. parameters["model_dir"])
    torch.save(parameters["model_dir"] .. "/word2index", word2index)
    torch.save(parameters["model_dir"] .. "/index2word", index2word)
    local table, table_size
        = BuildTable(vocab, word2index, parameters["unigram_model_power"], parameters["table_size"])
    local data = ReadData(parameters["corpus"])
    print(#data)
    local skip_gram, criterion = ConstructModel(vocab_size, parameters["dim"])
    ThreadedTrain(skip_gram, criterion, vocab, word2index, table, table_size, total_count, data, parameters)
end

return SkipGram

