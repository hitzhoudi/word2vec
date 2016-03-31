local threads = require 'threads'

local nthread = 4
local njob = 10
local msg = "hello from a satellite thread"

local pool = threads.Threads(
   nthread,
   function(threadid)
      print('starting a new thread/state number ' .. threadid)
      gmsg = msg -- get it the msg upvalue and store it in thread state
   end,

   function(threadid)
      local doct = "hello world"
      local words = doct
      function g(threadid)
         print('g(threadid): ' .. threadid)
         doct = doct .. doct
         print('doct: ' .. doct)
         return words
      end
   end

)


--local tt = t;

local jobdone = 0
for i=1,njob do
   pool:addjob(
      function()
         print(string.format('%s -- thread ID is %x', gmsg, __threadid))
         return g(__threadid)
      end,

      function(words)
         print("words from g(): " .. words)
         jobdone = jobdone + 1
      end
   )
end

pool:synchronize()

print(string.format('%d jobs done', jobdone))

pool:terminate()
