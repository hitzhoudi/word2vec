local threads = require 'threads'

local nthread = 4
local njob = 10
local msg = "hello from a satellite thread"

local t = torch.IntTensor(5)
t[1] = 3

local pool = threads.Threads(
   nthread,
   function(threadid)
      print('starting a new thread/state number ' .. threadid)
      gmsg = msg -- get it the msg upvalue and store it in thread state
   end,

   function(threadid)
      local m = t

      function g(threadid)
         print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh ' .. threadid)
         print('g(threadid): ' .. m[1])
      end
   end

)


--local tt = t;

local jobdone = 0
for i=1,njob do
   pool:addjob(
      function()
         print(string.format('%s -- thread ID is %x', gmsg, __threadid))
         g(__threadid)
         return __threadid
      end,

      function(id)
         t[1] = id
         print(string.format("task %d finished (ran on thread ID %x), t[1] == %d", i, id, t[1]))
         jobdone = jobdone + 1
      end
   )
end

pool:synchronize()

print(string.format('%d jobs done', jobdone))

pool:terminate()
