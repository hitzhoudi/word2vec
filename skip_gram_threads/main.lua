--dofile("skip_gram.lua")
require "nn"

config = {}
config.corpus = "../data/corpus.txt"    -- input file
config.window_size = 5                  -- maximum window size
config.neg_sample_number = 5            -- maximum negative sample number
config.learning_rate = 0.025            -- maximum learning rate
config.dim = 100                        -- dimensionality of word embedding
config.stream = 1                       -- stream mode
config.epochs = 10                      -- number of epochs
config.save_epochs = 1                  -- save rate
config.model_dir = "./model"            -- model direct
config.min_frequency = 10               -- minimum frequency
config.thread_number = 10               -- thread number
config.mode = "train"                   -- mode: train or test

cmd = torch.CmdLine()
cmd:option("-corpus", config.corpus)
cmd:option("-window_size", config.window_size)
cmd:option("-neg_sample_number", config.neg_sample_number)
cmd:option("-learning_rate", config.learning_rate)
cmd:option("-dim", config.dim)
cmd:option("-stream", config.stream)
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
parameters["stream"] = config.stream
parameters["epochs"] = config.epochs
parameters["save_epochs"] = config.save_epochs
parameters["model_dir"] = config.model_dir
parameters["min_frequency"] = config.min_frequency
parameters["vocab"] = {}
parameters["word2index"] = {}
parameters["index2word"] = {}
parameters["table_size"] = 1e8
parameters["table"] = {}
parameters["unigram_model_power"] = 0.75
parameters["label"] = torch.zeros(1 + parameters["neg_sample_number"]);
parameters["label"][1] = 1
parameters["thread_num"] = config.thread_number

local function Split(line, sep)
    local t = {}; local i = 1
    for word in string.gmatch(line, "%a+") do
        t[i] = word; i = i + 1
    end
    return t
end

local function BuildVocabulary()
    local start = sys.clock()
    local file = io.open(parameters["corpus"], "r")
    parameters["vocab"] = {}
    for line in file:lines() do
        for _, word in ipairs(Split(line)) do
            if parameters["vocab"][word] == nil then
                parameters["vocab"][word] = 1
            else
                parameters["vocab"][word] = parameters["vocab"][word] + 1
            end
        end
    end
    file.close()
    parameters["word2index"] = {}; parameters["index2word"] = {}
    for word, count in pairs(parameters["vocab"]) do
        if count >= parameters["min_frequency"] then
            parameters["index2word"][#parameters["index2word"] + 1] = word;
            parameters["word2index"][word] = #parameters["index2word"];
        else
            parameters["vocab"][word] = nil
        end
    end
    parameters["vocab_size"] = #parameters["index2word"]
    print(string.format("BuildVacabulary costs: %d", sys.clock() - start))
end

local function BuildTable()
    local start = sys.clock()
    local total_count_power = 0
    for _, count in pairs(parameters["vocab"]) do
        total_count_power = total_count_power + count^parameters["unigram_model_power"]
    end

    parameters["table"] = torch.IntTensor(parameters["table_size"])
    local accum_word_prob = 0; accum_word_index = 0
    for word, count in pairs(parameters["vocab"]) do
        accum_word_prob = accum_word_prob + count^parameters["unigram_model_power"] / total_count_power
        while (accum_word_index + 1) / parameters["table_size"] < accum_word_prob do
            accum_word_index = accum_word_index + 1
            parameters["table"][accum_word_index] = parameters["word2index"][word]
        end
    end
    parameters["table_size"] = accum_word_index
    print(string.format("BuildTable costs: %d", sys.clock() - start))
end

local function ConstructModel()
    local skip_gram = nn.Sequential()
    skip_gram:add(nn.ParallelTable())
    local word_vector = nn.LookupTable(parameters["vocab_size"], parameters["dim"])
    word_vector:reset(0.25);
    local context_vector = nn.LookupTable(parameters["vocab_size"], parameters["dim"])
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
    BuildVocabulary()
    BuildTable()
    local skip_gram, criterion = ConstructModel()
    local ThreadedTrain = require 'threadedtrain'
    ThreadedTrain(skip_gram, criterion, parameters)
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
