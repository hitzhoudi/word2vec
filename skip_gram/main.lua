dofile("skip_gram.lua")

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

model = SkipGram()
if config.mode == "train" then
    model:BuildVocabulary()
    model:BuildTable()
    model:ConstructModel()
    model:Train()
elseif  config.mode == "test" then
    model:PrintSimWords("free", 5)
else
    print("Error: mode should be train or test")
end
