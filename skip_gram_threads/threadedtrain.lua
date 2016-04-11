require("nn")
local Threads = require 'threads'

SkipGram = {}

local function Split(line, sep)
    if sep == nil then
        sep = "%s"
    end
    local t = {}; local i = 1
    for word in string.gmatch(line, "([^"..sep.."]+)") do
        t[i] = word; i = i + 1
    end
    return t
end

local function BuildVocabulary(corpus_path, min_frequency)
    local start = sys.clock()
    local file = io.open(corpus_path, "r")
    local vocab = {}
    for line in file:lines() do
        for _, word in ipairs(Split(line)) do
            if vocab[word] == nil then
                vocab[word] = 1
            else
                vocab[word] = vocab[word] + 1
            end
        end
    end
    file.close()
    local train_word_num = 0
    local word2index = {}
    local index2word = {}
    for word, count in pairs(vocab) do
        if count >= min_frequency then
            index2word[#index2word + 1] = word;
            word2index[word] = #index2word;
            train_word_num = train_word_num + count
        else
            vocab[word] = nil
        end
    end
    local vocab_size = #index2word
    print(string.format("BuildVacabulary costs: %d second(s)", sys.clock() - start))
    return vocab, vocab_size, word2index, index2word, train_word_num
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

local function ThreadedTrain(module, criterion, vocabulary, word2index, table, table_size, train_word_num, data, parameters)
    -- shared among threads

    Threads.serialization('threads.sharedserialize')
    local threads = Threads(
        parameters["thread_num"],
        function()
            require 'nn'
        end,
        function()
            local module = module:clone('weight', 'bias')
            local weights, dweights = module:parameters()
            local criterion = criterion:clone()
            local label = torch.zeros(1 + parameters["neg_sample_number"]); label[1] = 1
            local contexts = torch.IntTensor(1 + parameters["neg_sample_number"])
            local lr = parameters["lr"]

            function pass()
                local s = math.floor(#data / parameters["thread_num"] * (__threadid - 1) + 1)
                local t = math.floor(#data / parameters["thread_num"] * __threadid + 1)
                t = math.min(t, #data + 1)
                local count, iter = 0, 0
                local total_count, last_total_count = 0, 0

                while true do
                    -- update lr
                    if total_count - last_total_count >= 10000 then
                        lr = parameters["lr"] * (1 - total_count / (parameters["epochs"] * (t - s) + 1))
                        lr = math.max(lr, parameters["lr"] * 1e-3)
                        print(string.format("thread id %d learning rate %f", __threadid, lr))
                        last_total_count = total_count
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
                            local ran = (1 + math.sqrt(vocabulary[word] / (parameters["sample"] * train_word_num)))
                                * parameters["sample"] * train_word_num / vocabulary[word]
                            if ran > math.random() then 
                                buffer[#buffer + 1] = word
                            end
                        end 
                        count = count + 1
                        total_count = total_count + 1
                    end
                    -- train core
                    local err = 0
                    for i, word in ipairs(buffer) do
                        local wid = word2index[word]
                        local center_word = torch.IntTensor(1)
                        center_word[1] = wid
                        local cur_window_size = torch.random(parameters["window_size"])
                        for j = i - cur_window_size, i + cur_window_size do
                            if j >= 1 and j <= #buffer and j ~= i then
                                local cid = word2index[buffer[j]]
                                contexts[1] = cid
                                local n = 1
                                while n <= parameters["neg_sample_number"] do
                                    local nid = table[torch.random(table_size)]
                                    if nid ~= cid and nid ~= wid then
                                        contexts[n + 1] = nid
                                        n = n + 1
                                    end
                                end

                                local output = module:forward({center_word, contexts})
                                local gap = criterion:forward(output, label)
                                err = err + gap
                                module:zeroGradParameters()
                                module:backward({center_word, contexts}, criterion:backward(output, label))
                                weights[1][center_word[1]]:add(-lr, dweights[1][center_word[1]])
                                for k = 1, parameters["neg_sample_number"] do
                                    weights[2][contexts[k]]:add(-lr, dweights[2][contexts[k]])
                                end
                            end
                        end
                    end
                    print(string.format("thread id %d, error %f", __threadid, err))
                end
            end
        end
    )

    local start = sys.clock()
    local weights = module:parameters()
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
    local vocab, vocab_size, word2index, index2word, train_word_num
        = BuildVocabulary(parameters["corpus"], parameters["min_frequency"])
    print("Train Word Number:" .. train_word_num .. " Vocabulary Size:" .. vocab_size)
    os.execute("mkdir " .. parameters["model_dir"])
    torch.save(parameters["model_dir"] .. "/word2index", word2index)
    torch.save(parameters["model_dir"] .. "/index2word", index2word)
    local table, table_size
        = BuildTable(vocab, word2index, parameters["unigram_model_power"], parameters["table_size"])
    local data = ReadData(parameters["corpus"])
    local skip_gram, criterion = ConstructModel(vocab_size, parameters["dim"])
    ThreadedTrain(skip_gram, criterion, vocab, word2index, table, table_size, train_word_num, data, parameters)
    torch.save(parameters["model_dir"] .. "/skip_gram", skip_gram)
    torch.save(parameters["model_dir"] .. "/word2index", word2index)
    torch.save(parameters["model_dir"] .. "/index2word", index2word)
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

