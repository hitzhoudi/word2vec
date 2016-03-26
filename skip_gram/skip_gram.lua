require("sys")
require("nn")

do
local SkipGram = torch.class("SkipGram")

function SkipGram:__init()
    self.vocab = {}
    self.dim = 100
    self.word2index = {}
    self.index2word = {}
    self.table_size = 1e7
    self.table = {}
    self.unigram_model_power = 0.75
    self.learning_rate = 0.025
    self.window_size = 5
    self.neg_sample_number = 5
    self.label = torch.zeros(1 + self.neg_sample_number); self.label[1] = 1

end

function Split(line, sep)
    local t = {}; local i = 1
    for word in string.gmatch(line, "%a+") do
        t[i] = word; i = i + 1
    end
    return t
end

function SkipGram:BuildVocabulary(corpus)
    local start = sys.clock()
    local file = io.open(corpus, "r")
    self.vocab = {}
    for line in file:lines() do
        for _, word in ipairs(Split(line)) do
            if self.vocab[word] == nil then
                self.vocab[word] = 1
            else
                self.vocab[word] = self.vocab[word] + 1
            end
        end
    end
    file.close()
    self.word2index = {}; self.index2word = {}
    for word, count in pairs(self.vocab) do
        if count >= 10 then
            self.index2word[#self.index2word + 1] = word;
            self.word2index[word] = #self.index2word;
        else
            self.vocab[word] = nil
        end
    end
    self.vocab_size = #self.index2word
    print(string.format("BuildVacabulary costs: %d", sys.clock() - start))
end

function SkipGram:BuildTable()
    local start = sys.clock()
    local total_count_power = 0
    for _, count in pairs(self.vocab) do
        total_count_power = total_count_power + count^self.unigram_model_power
    end

    self.table = torch.IntTensor(self.table_size)
    local accum_word_prob = 0; accum_word_index = 0
    for word, count in pairs(self.vocab) do
        accum_word_prob = accum_word_prob + count^self.unigram_model_power / total_count_power
        while (accum_word_index + 1) / self.table_size < accum_word_prob do
            accum_word_index = accum_word_index + 1
            self.table[accum_word_index] = self.word2index[word]
        end
    end
    self.table_size = accum_word_index
    print(string.format("BuildTable costs: %d", sys.clock() - start))
end

function SkipGram:ConstructModel()
    self.skip_gram = nn.Sequential()
    self.skip_gram:add(nn.ParallelTable())
    self.word_vector = nn.LookupTable(self.vocab_size, self.dim)
    self.word_vector:reset(0.25);
    self.context_vector = nn.LookupTable(self.vocab_size, self.dim)
    self.context_vector:reset(0.25)
    self.skip_gram.modules[1]:add(self.word_vector)
    self.skip_gram.modules[1]:add(self.context_vector)
    self.skip_gram:add(nn.MM(false, true))
    self.skip_gram:add(nn.Sigmoid())
    self.criterion = nn.BCECriterion()
    -- TODO
end

function SkipGram:TrainUnit(word, contexts)
    local f = self.skip_gram:forward({word, contexts})
    local l = self.criterion:forward(f, self.label)
    local df_dl = self.criterion:backward(f, self.label)
    self.skip_gram:zeroGradParameters()
    self.skip_gram:backward({word, contexts}, df_dl)
    self.skip_gram:updateParameters(self.learning_rate)
end

function SkipGram:GenerateContexts(pos_context_id)
    local contexts = torch.IntTensor(1 + self.neg_sample_number)
    contexts[1] = pos_context_id
    local i = 0
    while i < self.neg_sample_number do
        local neg_context_id = self.table[torch.random(self.table_size)]
        if neg_context_id ~= pos_context_id then
            contexts[i + 2] = neg_context_id
            i = i + 1
        end
    end
    return contexts
end

function SkipGram:TrainStream()
    local start = sys.clock()
    local c = 0
    local file = io.open(corpus, "r")
    local center_word = torch.IntTensor(1)
    for line in file:lines() do
        local sentence = Split(line)
        for i, word in ipairs(sentence) do
            local word_idx = self.word2index[word]
            if word_idx ~= nil then
                center_word[1] = word_idx
                local cur_window_size = torch.random(self.window_size)
                for j = i - cur_window_size, i + cur_window_size do
                    local context = sentence[j]
                    if context ~= nil and j ~= i then
                        local context_id = self.word2index[context]
                        if context_id ~= nil then
                            local contexts = self:GenerateContexts(context_id)
                            self:TrainUnit(center_word, contexts)
                        end
                        c = c + 1
                        if c % 100000 == 0 then
                            print(string.format("Cost %d", sys.clock() - start))
                        end
                    end
                end
            end
        end
    end
end

function SkipGram:normalize()
    local m = self.word_vector.weight:double()
    local m_norm = torch.zeros(m:size())
    for i = 1, m:size(1) do
        m_norm[i] = m[i] / torch.norm(m[i])
    end
    return m_norm
end

function SkipGram:GetSimWords(w, k)
    if self.word_vector_norm == nil then
--        self.word_vector_norm = self:normalize()
        local m = self.word_vector.weight:double()
        local m_norm = torch.zeros(m:size())
        for i = 1, m:size(1) do
            m_norm[i] = m[i] / torch.norm(m[i])
        end
        self.word_vector_norm = m_norm
    end
    if type(w) == "string" then
        if self.word2index(w) == nil then
            print(string.format("%s does not exit in Vocabulary", w))
        else
            w = self.word_vecs_norm[self.word2index[w]]
            local similary = torch.mv(self.word_vector_norm, w)
            similary, idx = torch.sort(similary, 1, true)
            local results = {}
            for i = 1, k do
                results[i] = {self.index2word[idx], similary[i]}
            end
            return results
        end
    end
    return nil
end

function SkipGram:PrintSimWords(w, k)
    r = self.GetSimWords(w, k)
    for i = 1, k do
        print(string.format("%s, %f", r[i][1], r[i][2]))
    end
end

end

corpus = "./cankao/word2vec_torch/corpus.txt"
model = SkipGram()
model:BuildVocabulary(corpus)
model:BuildTable()
model:ConstructModel()
model:TrainStream()
model:PrintSimWords("china", 5)
