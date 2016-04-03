require('nn')

config = {}
config.corpus = "../data/corpus.txt"    -- input file
config.window_size = 5                  -- maximum window size
config.neg_sample_number = 5            -- maximum negative sample number
config.learning_rate = 0.025            -- maximum learning rate
config.dim = 100                        -- dimensionality of word embedding
config.epochs = 10                      -- number of epochs
config.save_epochs = 1                  -- save rate
config.model_dir = "./model"            -- model direct
config.min_frequency = 10               -- minimum frequency
config.thread_number = 2                -- thread number
config.mode = "train"                   -- mode: train or test

cmd = torch.CmdLine()
cmd:option("-corpus", config.corpus)
cmd:option("-window_size", config.window_size)
cmd:option("-neg_sample_number", config.neg_sample_number)
cmd:option("-learning_rate", config.learning_rate)
cmd:option("-dim", config.dim)
cmd:option("-epochs", config.epochs)
cmd:option("-save_epochs", config.save_epochs)
cmd:option("-model_dir", config.model_dir)
cmd:option("-min_frequency", config.min_frequency)
cmd:option("-thread_number", config.thread_number)
cmd:option("-mode", config.mode)

params = cmd:parse(arg)

for param, value in pairs(params) do
    config[param] = value
end

parameters = {}
parameters["corpus"] = config.corpus
parameters["window_size"] = config.window_size
parameters["neg_sample_number"] = config.neg_sample_number
parameters["learning_rate"] = config.learning_rate
parameters["dim"] = config.dim
parameters["epochs"] = config.epochs
parameters["save_epochs"] = config.save_epochs
parameters["model_dir"] = config.model_dir
parameters["min_frequency"] = config.min_frequency
parameters["thread_num"] = config.thread_number

parameters["vocab"] = {}
parameters["word2index"] = {}
parameters["index2word"] = {}
parameters["table_size"] = 1e8
parameters["table"] = {}
parameters["unigram_model_power"] = 0.75

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
    print(string.format("Vocabulary size is %d", vocab_size))
    return vocab, vocab_size, word2index, index2word
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
    --table_size = accum_word_index
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
    -- TODO

    return skip_gram, criterion
end

if config.mode == "train" then
    local vocab, vocab_size, word2index, index2word
        = BuildVocabulary(parameters["corpus"], parameters["min_frequency"])
    local table, table_size
        = BuildTable(vocab, word2index, parameters["unigram_model_power"], parameters["table_size"])
    local skip_gram, criterion = ConstructModel(vocab_size, parameters["dim"])
    local ThreadedTrain = require 'threadedtrain'
    ThreadedTrain(skip_gram, criterion, word2index, table, table_size, parameters)
end



--[[model = SkipGram()
if config.mode == "train" then
    BuildVocabulary()
    model:BuildTable()
    model:ConstructModel()
    model:Train()
elseif  config.mode == "test" then
    model:PrintSimWords("free", 5)
else
    print("Error: mode should be train or test")
end]]--
