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


function SkipGram:TrainMemory()
    local start = sys.clock()
    for i = 1, #self.train_words do
        self:TrainUnit(self.train_words[i], self.train_contexts[i])
        if i % 100000 == 0 then
            print(string.format("TrainMemory Cost %d", sys.clock() - start))
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
        self.word_vector_norm = self:normalize()
    end
    if type(w) == "string" then
        if self.word2index[w] == nil then
            print(string.format("%s does not exit in Vocabulary", w))
        else
            w = self.word_vector_norm[self.word2index[w]]
            local similary = torch.mv(self.word_vector_norm, w)
            similary, idx = torch.sort(similary, 1, true)
            local results = {}
            for i = 1, k do
                results[i] = {self.index2word[idx[i]], similary[i]}
            end
            return results
        end
    end
    return nil
end

function SkipGram:PrintSimWords(w, k)
    if self.skip_gram == nil then
        self.skip_gram = torch.load(self.model_dir .. "/skip_gram")
        self.word_vector = self.skip_gram.modules[1].modules[1]
        self.context_vector = self.skip_gram.modules[1].modules[2]
        self.word2index = torch.load(self.model_dir .. "/word2index")
        self.index2word = torch.load(self.model_dir .. "/index2word")
        print("Model SkipGram is loaded")
    end
    r = self:GetSimWords(w, k)
    if r == nil then
        return
    end
    for i = 1, k do
        print(string.format("%s, %f", r[i][1], r[i][2]))
    end
end

function SkipGram:Train()
    os.execute("mkdir " .. self.model_dir)
    for i = 1, self.epochs do
        if self.stream == 1 then
            self:TrainStream()
        else
            model:LoadData()
            self:TrainMemory()
        end
        if (i == 1) then
            torch.save(self.model_dir .. "/word2index", self.word2index)
            torch.save(self.model_dir .. "/index2word", self.index2word)
        end
        if (i % self.save_epochs == 0) then
            torch.save(self.model_dir .. "/skip_gram_epoch_" .. tostring(i), self.skip_gram)
        end
        print(string.format("%d epoch(s) is done", i))
    end
    torch.save(self.model_dir .. "/epoch_" .. tostring(self.epochs), self.skip_gram)
end
