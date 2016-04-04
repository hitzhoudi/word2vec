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

local function ThreadedTrain(module, criterion, vocabulary, word2index, table, table_size, total_count, parameters)
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
            local label = torch.zeros(1 + parameters["neg_sample_number"])
            label[1] = 1

            function pass(data, learning_rate)
                local output = module:forward(data)
                local gap = criterion:forward(output, label)
                module:zeroGradParameters()
                module:backward(data, criterion:backward(output, label))
                weights[1][data[1][1]]:add(-learning_rate, dweights[1][data[1][1]])
                for i = 1, parameters["neg_sample_number"] do
                    weights[2][data[2][i]]:add(-learning_rate, dweights[2][data[2][i]])
                end
                return gap
            end
        end

    )

    local function GenerateContexts(pos_context_id)
        local contexts = torch.IntTensor(1 + parameters["neg_sample_number"])
        contexts[1] = pos_context_id
        local i = 0
        while i < parameters["neg_sample_number"] do
            local neg_context_id = table[torch.random(table_size)]
            if neg_context_id ~= pos_context_id then
                contexts[i + 2] = neg_context_id
                i = i + 1
            end
        end
        return contexts
    end

    local start = sys.clock()
    local weights = module:parameters()

    local lr = parameters["learning_rate"]
    local cur_word_count = 0
    for iter = 1, parameters["epochs"] do
        local file = io.open(parameters["corpus"], "r")
        local c = 0
        local t_err = 0
        for line in file:lines() do
            local sentence = Split(line)
            for i, word in ipairs(sentence) do
                local word_idx = word2index[word]
                if word_idx ~= nil then
                    cur_word_count = cur_word_count + 1
                    -- update learning rate
                    if cur_word_count % 10000 == 0 and cur_word_count > 0 then
                        lr = parameters["learning_rate"] * (1 - cur_word_count / (1 + iter * total_count))
                        lr = math.max(parameters["min_learning_rate"], lr)
                    end
                    -- The subsampling randomly discards frequent words while keeping the ranking same
                    local ran = (1 + math.sqrt(vocabulary[word] / (parameters["sample"] * total_count)))
                            * parameters["sample"] * total_count / vocabulary[word]
                    print("ran " .. ran .. "vacabulary[word] " .. vocabulary[word])
                    if ran < math.random() then 
                        local center_word = torch.IntTensor(1)
                        center_word[1] = word_idx
                        local cur_window_size = torch.random(parameters["window_size"])
                        for j = i - cur_window_size, i + cur_window_size do
                            local context = sentence[j]
                            if context ~= nil and j ~= i then
                                local context_id = word2index[context]
                                if context_id ~= nil then
                                    local contexts = GenerateContexts(context_id)
                                    threads:addjob(
                                        function()
                                            return pass({center_word, contexts}, lr)
                                        end,
                                        function(gap)
                                            t_err = t_err + gap
                                        end
                                    )
                                end
                            end
                        end
                    end
                end
                c = c + 1
                if (c % 10000 == 0) then
                    print(string.format(
                        "Epoch[%d] Part[%d] AccumErrorRate[%f] LR[%f] Cost[%d second(s)]",
                        iter, c / 100000, t_err / c, lr, sys.clock() - start))
                    start = sys.clock()
                end
            end
        end
        file.close()
        if (iter % parameters["save_epochs"] == 0) then
            torch.save(parameters["model_dir"] .. "/skip_gram_epoch_" .. tostring(iter), module)
        end
        threads:synchronize()
    end
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
    local skip_gram, criterion = ConstructModel(vocab_size, parameters["dim"])
    ThreadedTrain(skip_gram, criterion, vocab, word2index, table, table_size, total_count, parameters)
end

function Normalize(word_vector)
    local m = word_vector.weight:double()
    local m_norm = torch.zeros(m:size())
    for i = 1, m:size(1) do
        m_norm[i] = m[i] / torch.norm(m[i])
    end
    return m_norm
end

function LoadModel(parameters)
    if SkipGram.skip_gram == nil
        or SkipGram.word2index == nil
        or SkipGram.index2word == nil then
        SkipGram.skip_gram = torch.load(parameters["model_dir"] .. "/skip_gram")
        SkipGram.word2index = torch.load(parameters["model_dir"] .. "/word2index")
        SkipGram.index2word = torch.load(parameters["model_dir"] .. "/index2word")
        word_vector = SkipGram.skip_gram.modules[1].modules[1]
        SkipGram.word_vector_norm = Normalize(word_vector)
    end
end

function SkipGram.GetSimWords(w, k, parameters)
    LoadModel(parameters)
    if type(w) == "string" then
        if SkipGram.word2index[w] == nil then
            print(string.format("%s does not exit in Vocabulary", w))
        else
            w = SkipGram.word_vector_norm[SkipGram.word2index[w]]
            local similary = torch.mv(SkipGram.word_vector_norm, w)
            similary, idx = torch.sort(similary, 1, true)
            local results = {}
            for i = 1, k do
                results[i] = {SkipGram.index2word[idx[i]], similary[i]}
            end
            return results
        end
    end
    return nil
end

return SkipGram
