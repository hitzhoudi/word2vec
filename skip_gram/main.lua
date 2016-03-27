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
config.train = 1                        -- train
config.test = 1                         -- test

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
cmd:option("-train", config.train)
cmd:option("-test", config.test)

params = cmd:parse(arg)

for param, value in pairs(params) do
    config[param] = value
end

if config.train == 1 then
    model = SkipGram()
    model:BuildVocabulary()
    model:BuildTable()
    model:ConstructModel()
    model:Train()
end
if config.test == 1 then
    model:PrintSimWords("people", 5)
end

