- sqlang
  - 初始化过程
  - 用户请求响应过程
    - /v1/chat/completions响应过程
      1. 用户发送请求到inference server. (sglang\python\sglang\srt\entrypoints\http_server.py)
      2. inference server的接口 "/v1/completions" 收到请求后，调用函数 openai_v1_completions() 进行处理。
         - openai_v1_completions
           - raw_request.app.state.openai_serving_completion.handle_request()
             - _validate_request    -- openAI协议级校验。
             - _convert_to_internal_request    -- 把用户请求转换成内部请求对象。
             - 判断响应是否需要流式传输
               - 需要，走 _handle_streaming_request() 函数进行处理。
                 - _generate_chat_stream()    -- 创建真正负责生成 OpenAI SSE 数据块的异步生成器。
                   - self.tokenizer_manager.generate_request()    -- 这个函数会调用 tokenizer_manager 内部的 generate() 函数，进行真正的 token 生成过程。
                     - self._tokenize_texts()    -- 把输入文本转换成 token ids 的过程。
                       - await self.async_dynamic_batch_tokenizer.encode()    -- sglang\python\sglang\srt\managers\async_dynamic_batch_tokenizer.py
                         - self._ensure_initialized()    -- 确保异步批量 tokenizer 已经初始化完成，准备好进行编码。此步骤创建了异步队列。
                           - asyncio.create_task(self._dynamic_batch_loop())    -- 启动一个后台任务，持续运行 _dynamic_batch_loop()，这个循环会不断从队列中取出待编码的文本进行批量处理。
                         - await self._queue.put()    -- 把待编码的文本和一个 future 对象放入队列中，等待后台任务处理。
                         - _dynamic_batch_loop() 循环等待函数，从内部_queue中获得到promot进行处理。然后把结果设置到对应的 future 对象中，完成异步编码过程。
                           - await self._queue.get()    -- 阻塞，获得promot。在等待窗口时间内进行合批，即尽可能多的收集待token化的文本。
                           - await self._process_dynamic_batch()    -- 对收集到的文本进行批量token化处理。
                             - 调用tokenizer把用户输入转换为token序列。
                             - fut.set_result(res)    -- 把token化后的token ids写回到future中返回。
                 - prepend_first_chunk()    -- i盗用上一步生成的generator，进行异步的token生成的过程。
               - 不需要，走 _handle_non_streaming_request() 函数进行处理。

# 模块说明
## inference server
- 源代码路径
  - E:\codex_home\code\sglang\python\sglang\srt\entrypoints\http_server.py