--dofile("skip_gram.lua")

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
parameters["table_size"] = 1e7
parameters["table"] = {}
parameters["unigram_model_power"] = 0.75
parameters["label"] = torch.zeros(1 + parameters["neg_sample_number"]);
parameters["label"][1] = 1

function Split(line, sep)
    local t = {}; local i = 1
    for word in string.gmatch(line, "%a+") do
        t[i] = word; i = i + 1
    end
    return t
end

function BuildVocabulary()
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

if config.mode == "train" then
    BuildVocabulary()
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
