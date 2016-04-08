config = {}
config.corpus = "../data/corpus.txt"    -- input file
config.window_size = 5                  -- maximum window size
config.neg_sample_number = 5            -- maximum negative sample number
config.learning_rate = 0.025            -- maximum learning rate
config.min_learning_rate = 0.001        -- minimum learning rate
config.dim = 100                        -- dimensionality of word embedding
config.epochs = 1                       -- number of epochs
config.save_epochs = 1                  -- save rate
config.model_dir = "./model"            -- model direct
config.min_frequency = 10               -- minimum frequency
config.thread_number = 10               -- thread number
config.mode = "train"                   -- mode: train or test
config.sample = 1.0e-4

cmd = torch.CmdLine()
cmd:option("-corpus", config.corpus)
cmd:option("-window_size", config.window_size)
cmd:option("-neg_sample_number", config.neg_sample_number)
cmd:option("-learning_rate", config.learning_rate)
cmd:option("-min_learning_rate", config.min_learning_rate)
cmd:option("-dim", config.dim)
cmd:option("-epochs", config.epochs)
cmd:option("-save_epochs", config.save_epochs)
cmd:option("-model_dir", config.model_dir)
cmd:option("-min_frequency", config.min_frequency)
cmd:option("-thread_number", config.thread_number)
cmd:option("-mode", config.mode)
cmd:option("-sample", config.sample)

params = cmd:parse(arg)

for param, value in pairs(params) do
    config[param] = value
end

parameters = {}
parameters["corpus"] = config.corpus
parameters["window_size"] = config.window_size
parameters["neg_sample_number"] = config.neg_sample_number
parameters["learning_rate"] = config.learning_rate
parameters["min_learning_rate"] = config.min_learning_rate
parameters["dim"] = config.dim
parameters["epochs"] = config.epochs
parameters["save_epochs"] = config.save_epochs
parameters["model_dir"] = config.model_dir
parameters["min_frequency"] = config.min_frequency
parameters["thread_num"] = config.thread_number
parameters["sample"] = config.sample

--parameters["vocab"] = {}
--parameters["word2index"] = {}
--parameters["index2word"] = {}
parameters["table_size"] = 1e7
--parameters["table"] = {}
parameters["unigram_model_power"] = 0.75

local SkipGram = require('threadedtrain')
if config.mode == "train" then
    SkipGram.Train()
elseif  config.mode == "test" then
    print(SkipGram.GetSimWords("america", 100, parameters))
end
