# ROCm性能数据（RPD）到Chrome追踪格式的转换工具
# 将ROCm profiler生成的SQLite数据库转换为Chrome兼容的追踪JSON格式
# 源自：https://raw.githubusercontent.com/ROCm/rocmProfileData/refs/heads/master/tools/rpd2tracing.py
# https://raw.githubusercontent.com/ROCm/rocmProfileData/refs/heads/master/tools/rpd2tracing.py  # 原始来源URL
# commit 92d13a08328625463e9ba944cece82fc5eea36e6  # 源提交哈希
def rpd_to_chrome_trace(  # 将RPD文件转换为Chrome追踪格式
    input_rpd, output_json=None, start="0%", end="100%", format="object"  # 输入RPD文件、输出JSON路径、时间范围、输出格式
):
    import gzip  # 导入gzip压缩模块
    import sqlite3  # 导入SQLite数据库模块

    if output_json is None:  # 如果未指定输出路径
        import pathlib  # 导入路径处理模块

        output_json = pathlib.PurePath(input_rpd).with_suffix(".trace.json.gz")  # 自动生成输出路径

    connection = sqlite3.connect(input_rpd)  # 连接到RPD SQLite数据库

    outfile = gzip.open(output_json, "wt", encoding="utf-8")  # 以文本模式打开gzip输出文件

    if format == "object":  # 如果输出格式为对象
        outfile.write('{"traceEvents": ')  # 写入traceEvents键

    outfile.write("[ {}\n")  # 写入事件数组起始

    for row in connection.execute("select distinct gpuId from rocpd_op"):  # 查询所有不同的GPU ID
        try:
            outfile.write(  # 写入进程名称元数据
                ',{"name": "process_name", "ph": "M", "pid":"%s","args":{"name":"%s"}}\n'
                % (row[0], "GPU" + str(row[0]))  # 使用GPU ID作为进程名
            )
            outfile.write(  # 写入进程排序索引元数据
                ',{"name": "process_sort_index", "ph": "M", "pid":"%s","args":{"sort_index":"%s"}}\n'
                % (row[0], row[0] + 1000000)  # 排序索引偏移1000000以确保GPU进程排在后面
            )
        except ValueError:
            outfile.write("")  # 忽略值错误

    for row in connection.execute("select distinct pid, tid from rocpd_api"):  # 查询所有不同的API线程
        try:
            outfile.write(  # 写入线程名称元数据
                ',{"name":"thread_name","ph":"M","pid":"%s","tid":"%s","args":{"name":"%s"}}\n'
                % (row[0], row[1], "Hip " + str(row[1]))  # 使用Hip前缀命名线程
            )
            outfile.write(  # 写入线程排序索引元数据
                ',{"name":"thread_sort_index","ph":"M","pid":"%s","tid":"%s","args":{"sort_index":"%s"}}\n'
                % (row[0], row[1], row[1] * 2)  # 排序索引为线程ID的2倍
            )
        except ValueError:
            outfile.write("")  # 忽略值错误

    try:
        # FIXME - these aren't rendering correctly in chrome://tracing  # 修复：这些在chrome://tracing中无法正确渲染
        for row in connection.execute("select distinct pid, tid from rocpd_hsaApi"):  # 查询所有不同的HSA API线程
            try:
                outfile.write(  # 写入HSA线程名称元数据
                    ',{"name":"thread_name","ph":"M","pid":"%s","tid":"%s","args":{"name":"%s"}}\n'
                    % (row[0], row[1], "HSA " + str(row[1]))  # 使用HSA前缀命名线程
                )
                outfile.write(  # 写入HSA线程排序索引元数据
                    ',{"name":"thread_sort_index","ph":"M","pid":"%s","tid":"%s","args":{"sort_index":"%s"}}\n'
                    % (row[0], row[1], row[1] * 2 - 1)  # 排序索引为线程ID的2倍减1
                )
            except ValueError:
                outfile.write("")  # 忽略值错误
    except:
        pass  # 忽略HSA表不存在的异常

    rangeStringApi = ""  # API时间范围过滤字符串
    rangeStringOp = ""  # 操作时间范围过滤字符串
    rangeStringMonitor = ""  # 监控时间范围过滤字符串
    min_time = connection.execute("select MIN(start) from rocpd_api;").fetchall()[0][0]  # 查询API最早开始时间
    max_time = connection.execute("select MAX(end) from rocpd_api;").fetchall()[0][0]  # 查询API最晚结束时间
    if min_time is None:  # 如果没有时间数据
        raise Exception("Trace file is empty.")  # 抛出追踪文件为空异常

    print("Timestamps:")  # 打印时间戳信息
    print(f"\t    first: \t{min_time/1000} us")  # 打印最早时间
    print(f"\t     last: \t{max_time/1000} us")  # 打印最晚时间
    print(f"\t duration: \t{(max_time-min_time) / 1000000000} seconds")  # 打印总持续时间

    start_time = min_time / 1000  # 计算起始时间（微秒）
    end_time = max_time / 1000  # 计算结束时间（微秒）

    if start:  # 如果指定了起始位置
        if "%" in start:  # 如果起始位置是百分比
            start_time = (
                (max_time - min_time) * (int(start.replace("%", "")) / 100) + min_time
            ) / 1000  # 按百分比计算起始时间
        else:
            start_time = int(start)  # 直接使用指定的起始时间
        rangeStringApi = "where rocpd_api.start/1000 >= %s" % (start_time)  # 构造API起始过滤条件
        rangeStringOp = "where rocpd_op.start/1000 >= %s" % (start_time)  # 构造操作起始过滤条件
        rangeStringMonitor = "where start/1000 >= %s" % (start_time)  # 构造监控起始过滤条件
    if end:  # 如果指定了结束位置
        if "%" in end:  # 如果结束位置是百分比
            end_time = (
                (max_time - min_time) * (int(end.replace("%", "")) / 100) + min_time
            ) / 1000  # 按百分比计算结束时间
        else:
            end_time = int(end)  # 直接使用指定的结束时间

        rangeStringApi = (
            rangeStringApi + " and rocpd_api.start/1000 <= %s" % (end_time)  # 添加API结束过滤条件
            if start != None  # 如果有起始条件，使用AND连接
            else "where rocpd_api.start/1000 <= %s" % (end_time)  # 否则使用WHERE
        )
        rangeStringOp = (
            rangeStringOp + " and rocpd_op.start/1000 <= %s" % (end_time)  # 添加操作结束过滤条件
            if start != None  # 如果有起始条件，使用AND连接
            else "where rocpd_op.start/1000 <= %s" % (end_time)  # 否则使用WHERE
        )
        rangeStringMonitor = (
            rangeStringMonitor + " and start/1000 <= %s" % (end_time)  # 添加监控结束过滤条件
            if start != None  # 如果有起始条件，使用AND连接
            else "where start/1000 <= %s" % (end_time)  # 否则使用WHERE
        )

    print("\nFilter: %s" % (rangeStringApi))  # 打印过滤条件
    print(f"Output duration: {(end_time-start_time)/1000000} seconds")  # 打印输出持续时间

    # Output Ops  # 输出GPU操作事件

    for row in connection.execute(  # 查询GPU操作事件
        "select A.string as optype, B.string as description, gpuId, queueId, rocpd_op.start/1000.0, (rocpd_op.end-rocpd_op.start) / 1000.0 from rocpd_op INNER JOIN rocpd_string A on A.id = rocpd_op.opType_id INNER Join rocpd_string B on B.id = rocpd_op.description_id %s"
        % (rangeStringOp)  # 应用时间范围过滤
    ):
        try:
            name = row[0] if len(row[1]) == 0 else row[1]  # 使用描述作为名称，无描述则使用操作类型
            outfile.write(  # 写入GPU操作事件
                ',{"pid":"%s","tid":"%s","name":"%s","ts":"%s","dur":"%s","ph":"X","args":{"desc":"%s"}}\n'
                % (row[2], row[3], name, row[4], row[5], row[0])  # GPU ID、队列ID、名称、时间戳、持续时间、描述
            )
        except ValueError:
            outfile.write("")  # 忽略值错误

    # Output Graph executions on GPU  # 输出GPU上的图执行事件
    try:
        for row in connection.execute(  # 查询图执行事件
            "select graphExec, gpuId, queueId, min(start)/1000.0, (max(end)-min(start))/1000.0, count(*) from rocpd_graphLaunchapi A join rocpd_api_ops B on B.api_id = A.api_ptr_id join rocpd_op C on C.id = B.op_id %s group by api_ptr_id"
            % (rangeStringMonitor)  # 应用时间范围过滤
        ):
            try:
                outfile.write(  # 写入图执行事件
                    ',{"pid":"%s","tid":"%s","name":"%s","ts":"%s","dur":"%s","ph":"X","args":{"kernels":"%s"}}\n'
                    % (row[1], row[2], f"Graph {row[0]}", row[3], row[4], row[5])  # GPU ID、队列ID、图名称、时间戳、持续时间、内核数
                )
            except ValueError:
                outfile.write("")  # 忽略值错误
    except:
        pass  # 忽略图表不存在的异常

    # Output apis  # 输出API调用事件
    for row in connection.execute(  # 查询API调用事件
        "select A.string as apiName, B.string as args, pid, tid, rocpd_api.start/1000.0, (rocpd_api.end-rocpd_api.start) / 1000.0, (rocpd_api.end != rocpd_api.start) as has_duration from rocpd_api INNER JOIN rocpd_string A on A.id = rocpd_api.apiName_id INNER Join rocpd_string B on B.id = rocpd_api.args_id %s order by rocpd_api.id"
        % (rangeStringApi)  # 应用时间范围过滤
    ):
        try:
            if row[0] == "UserMarker":  # 如果是用户标记事件
                if row[6] == 0:  # instantanuous "mark" messages  # 瞬时标记消息（无持续时间）
                    outfile.write(  # 写入瞬时事件
                        ',{"pid":"%s","tid":"%s","name":"%s","ts":"%s","ph":"i","s":"p","args":{"desc":"%s"}}\n'
                        % (
                            row[2],  # 进程ID
                            row[3],  # 线程ID
                            row[1].replace('"', ""),  # 名称（去除引号）
                            row[4],  # 时间戳
                            row[1].replace('"', ""),  # 描述（去除引号）
                        )
                    )
                else:  # 有持续时间的用户标记
                    outfile.write(  # 写入持续事件
                        ',{"pid":"%s","tid":"%s","name":"%s","ts":"%s","dur":"%s","ph":"X","args":{"desc":"%s"}}\n'
                        % (
                            row[2],  # 进程ID
                            row[3],  # 线程ID
                            row[1].replace('"', ""),  # 名称（去除引号）
                            row[4],  # 时间戳
                            row[5],  # 持续时间
                            row[1].replace('"', ""),  # 描述（去除引号）
                        )
                    )
            else:  # 非用户标记的API调用
                outfile.write(  # 写入API持续事件
                    ',{"pid":"%s","tid":"%s","name":"%s","ts":"%s","dur":"%s","ph":"X","args":{"desc":"%s"}}\n'
                    % (
                        row[2],  # 进程ID
                        row[3],  # 线程ID
                        row[0],  # API名称
                        row[4],  # 时间戳
                        row[5],  # 持续时间
                        row[1].replace('"', "").replace("\t", ""),  # 参数描述（去除引号和制表符）
                    )
                )
        except ValueError:
            outfile.write("")  # 忽略值错误

    # Output api->op linkage  # 输出API到操作的链接事件
    for row in connection.execute(  # 查询API到操作的关联
        "select rocpd_api_ops.id, pid, tid, gpuId, queueId, rocpd_api.end/1000.0 - 2, rocpd_op.start/1000.0 from rocpd_api_ops INNER JOIN rocpd_api on rocpd_api_ops.api_id = rocpd_api.id INNER JOIN rocpd_op on rocpd_api_ops.op_id = rocpd_op.id %s"
        % (rangeStringApi)  # 应用时间范围过滤
    ):
        try:
            fromtime = row[5] if row[5] < row[6] else row[6]  # 选择较早的时间作为流起始时间
            outfile.write(  # 写入流起始事件
                ',{"pid":"%s","tid":"%s","cat":"api_op","name":"api_op","ts":"%s","id":"%s","ph":"s"}\n'
                % (row[1], row[2], fromtime, row[0])  # 进程ID、线程ID、时间、关联ID
            )
            outfile.write(  # 写入流结束事件
                ',{"pid":"%s","tid":"%s","cat":"api_op","name":"api_op","ts":"%s","id":"%s","ph":"f", "bp":"e"}\n'
                % (row[3], row[4], row[6], row[0])  # GPU ID、队列ID、操作时间、关联ID
            )
        except ValueError:
            outfile.write("")  # 忽略值错误

    try:
        for row in connection.execute(  # 查询HSA API调用事件
            "select A.string as apiName, B.string as args, pid, tid, rocpd_hsaApi.start/1000.0, (rocpd_hsaApi.end-rocpd_hsaApi.start) / 1000.0 from rocpd_hsaApi INNER JOIN rocpd_string A on A.id = rocpd_hsaApi.apiName_id INNER Join rocpd_string B on B.id = rocpd_hsaApi.args_id %s order by rocpd_hsaApi.id"
            % (rangeStringApi)  # 应用时间范围过滤
        ):
            try:
                outfile.write(  # 写入HSA API持续事件
                    ',{"pid":"%s","tid":"%s","name":"%s","ts":"%s","dur":"%s","ph":"X","args":{"desc":"%s"}}\n'
                    % (
                        row[2],  # 进程ID
                        row[3] + 1,  # 线程ID偏移1以避免与Hip线程重叠
                        row[0],  # API名称
                        row[4],  # 时间戳
                        row[5],  # 持续时间
                        row[1].replace('"', ""),  # 参数描述（去除引号）
                    )
                )
            except ValueError:
                outfile.write("")  # 忽略值错误
    except:
        pass  # 忽略HSA表不存在的异常

    #  # 注释块
    # Counters  # 计数器事件
    #  # 注释块

    # Counters should extend to the last event in the trace.  This means they need to have a value at Tend.  # 计数器应扩展到追踪中的最后一个事件，需要在Tend处有值
    # Figure out when that is  # 确定结束时间

    T_end = 0  # 初始化结束时间
    for row in connection.execute(  # 查询所有事件的最晚结束时间
        "SELECT max(end)/1000 from (SELECT end from rocpd_api UNION ALL SELECT end from rocpd_op)"
    ):
        T_end = int(row[0])  # 更新结束时间
    if end:  # 如果指定了结束位置
        T_end = end_time  # 使用指定的结束时间

    # Loop over GPU for per-gpu counters  # 遍历GPU，生成每个GPU的计数器
    gpuIdsPresent = []  # 存在的GPU ID列表
    for row in connection.execute("SELECT DISTINCT gpuId FROM rocpd_op"):  # 查询所有不同的GPU ID
        gpuIdsPresent.append(row[0])  # 添加到列表

    for gpuId in gpuIdsPresent:  # 遍历每个GPU
        # print(f"Creating counters for: {gpuId}")  # 调试：打印正在创建计数器的GPU

        # Create the queue depth counter  # 创建队列深度计数器
        depth = 0  # 初始队列深度
        idle = 1  # 初始空闲状态
        for row in connection.execute(  # 查询队列深度变化事件
            'select * from (select rocpd_api.start/1000.0 as ts, "1" from rocpd_api_ops INNER JOIN rocpd_api on rocpd_api_ops.api_id = rocpd_api.id INNER JOIN rocpd_op on rocpd_api_ops.op_id = rocpd_op.id AND rocpd_op.gpuId = %s %s UNION ALL select rocpd_op.end/1000.0, "-1" from rocpd_api_ops INNER JOIN rocpd_api on rocpd_api_ops.api_id = rocpd_api.id INNER JOIN rocpd_op on rocpd_api_ops.op_id = rocpd_op.id AND rocpd_op.gpuId = %s %s) order by ts'
            % (gpuId, rangeStringOp, gpuId, rangeStringOp)  # 按GPU ID和时间范围过滤
        ):
            try:
                if idle and int(row[1]) > 0:  # 如果从空闲变为忙碌
                    idle = 0  # 设置为忙碌状态
                    outfile.write(  # 写入空闲计数器事件
                        ',{"pid":"%s","name":"Idle","ph":"C","ts":%s,"args":{"idle":%s}}\n'
                        % (gpuId, row[0], idle)  # GPU ID、时间戳、空闲状态
                    )
                if depth == 1 and int(row[1]) < 0:  # 如果队列深度从1变为0
                    idle = 1  # 设置为空闲状态
                    outfile.write(  # 写入空闲计数器事件
                        ',{"pid":"%s","name":"Idle","ph":"C","ts":%s,"args":{"idle":%s}}\n'
                        % (gpuId, row[0], idle)  # GPU ID、时间戳、空闲状态
                    )
                depth = depth + int(row[1])  # 更新队列深度
                outfile.write(  # 写入队列深度计数器事件
                    ',{"pid":"%s","name":"QueueDepth","ph":"C","ts":%s,"args":{"depth":%s}}\n'
                    % (gpuId, row[0], depth)  # GPU ID、时间戳、队列深度
                )
            except ValueError:
                outfile.write("")  # 忽略值错误
        if T_end > 0:  # 如果有结束时间
            outfile.write(  # 写入结束时的空闲计数器值
                ',{"pid":"%s","name":"Idle","ph":"C","ts":%s,"args":{"idle":%s}}\n'
                % (gpuId, T_end, idle)  # GPU ID、结束时间、空闲状态
            )
            outfile.write(  # 写入结束时的队列深度计数器值
                ',{"pid":"%s","name":"QueueDepth","ph":"C","ts":%s,"args":{"depth":%s}}\n'
                % (gpuId, T_end, depth)  # GPU ID、结束时间、队列深度
            )

    # Create SMI counters  # 创建SMI（系统管理接口）计数器
    try:
        for row in connection.execute(  # 查询SMI监控数据
            "select deviceId, monitorType, start/1000.0, value from rocpd_monitor %s"
            % (rangeStringMonitor)  # 应用时间范围过滤
        ):
            outfile.write(  # 写入SMI计数器事件
                ',{"pid":"%s","name":"%s","ph":"C","ts":%s,"args":{"%s":%s}}\n'
                % (row[0], row[1], row[2], row[1], row[3])  # 设备ID、监控类型、时间戳、值
            )
        # Output the endpoints of the last range  # 输出最后一个范围的端点值
        for row in connection.execute(  # 查询每种监控类型的最后值
            "select distinct deviceId, monitorType, max(end)/1000.0, value from rocpd_monitor %s group by deviceId, monitorType"
            % (rangeStringMonitor)  # 应用时间范围过滤
        ):
            outfile.write(  # 写入SMI计数器端点值
                ',{"pid":"%s","name":"%s","ph":"C","ts":%s,"args":{"%s":%s}}\n'
                % (row[0], row[1], row[2], row[1], row[3])  # 设备ID、监控类型、结束时间、值
            )
    except:
        print("Did not find SMI data")  # 打印未找到SMI数据

    # Create the (global) memory counter  # 创建全局内存计数器（已注释掉）
    """  # 多行注释开始
    sizes = {}    # address -> size
    totalSize = 0
    exp = re.compile("^ptr\((.*)\)\s+size\((.*)\)$")
    exp2 = re.compile("^ptr\((.*)\)$")
    for row in connection.execute("SELECT rocpd_api.end/1000.0 as ts, B.string, '1'  FROM rocpd_api INNER JOIN rocpd_string A ON A.id=rocpd_api.apiName_id INNER JOIN rocpd_string B ON B.id=rocpd_api.args_id WHERE A.string='hipFree' UNION ALL SELECT rocpd_api.start/1000.0, B.string, '0' FROM rocpd_api INNER JOIN rocpd_string A ON A.id=rocpd_api.apiName_id INNER JOIN rocpd_string B ON B.id=rocpd_api.args_id WHERE A.string='hipMalloc' ORDER BY ts asc"):
        try:
            if row[2] == '0':  #malloc
                m = exp.match(row[1])
                if m:
                    size = int(m.group(2), 16)
                    totalSize = totalSize + size
                    sizes[m.group(1)] = size
                    outfile.write(',{"pid":"0","name":"Allocated Memory","ph":"C","ts":%s,"args":{"depth":%s}}\n'%(row[0],totalSize))
            else:              #free
                m = exp2.match(row[1])
                if m:
                    try:    # Sometimes free addresses are not valid or listed
                        size = sizes[m.group(1)]
                        sizes[m.group(1)] = 0
                        totalSize = totalSize - size;
                        outfile.write(',{"pid":"0","name":"Allocated Memory","ph":"C","ts":%s,"args":{"depth":%s}}\n'%(row[0],totalSize))
                    except KeyError:
                        pass
        except ValueError:
            outfile.write("")
    if T_end > 0:
        outfile.write(',{"pid":"0","name":"Allocated Memory","ph":"C","ts":%s,"args":{"depth":%s}}\n'%(T_end,totalSize))
    """  # 多行注释结束

    # Create "faux calling stack frame" on gpu ops traceS  # 在GPU操作追踪上创建伪调用栈帧
    stacks = {}  # Call stacks built from UserMarker entres.     Key is 'pid,tid'  # 从UserMarker条目构建的调用栈，键为'pid,tid'
    currentFrame = {}  # "Current GPU frame" (id, name, start, end).    Key is 'pid,tid'  # 当前GPU帧（id、名称、起始、结束），键为'pid,tid'

    class GpuFrame:  # GPU帧数据类
        def __init__(self):  # 初始化GPU帧
            self.id = 0  # 帧ID
            self.name = ""  # 帧名称
            self.start = 0  # 帧起始时间
            self.end = 0  # 帧结束时间
            self.gpus = []  # 帧涉及的GPU列表
            self.totalOps = 0  # 帧中的操作总数

    # FIXME: include 'start' (in ns) so we can ORDER BY it and break ties?  # 修复：包含'start'（纳秒）以便排序和打破平局
    for row in connection.execute(  # 查询所有帧起始、结束和操作事件
        "SELECT '0', start/1000.0, pid, tid, B.string as label, '','','', '' from rocpd_api INNER JOIN rocpd_string A on A.id = rocpd_api.apiName_id AND A.string = 'UserMarker' INNER JOIN rocpd_string B on B.id = rocpd_api.args_id AND rocpd_api.start/1000.0 != rocpd_api.end/1000.0 %s UNION ALL SELECT '1', end/1000.0, pid, tid, B.string as label, '','','', '' from rocpd_api INNER JOIN rocpd_string A on A.id = rocpd_api.apiName_id AND A.string = 'UserMarker' INNER JOIN rocpd_string B on B.id = rocpd_api.args_id AND rocpd_api.start/1000.0 != rocpd_api.end/1000.0 %s UNION ALL SELECT '2', rocpd_api.start/1000.0, pid, tid, '' as label, gpuId, queueId, rocpd_op.start/1000.0, rocpd_op.end/1000.0 from rocpd_api_ops INNER JOIN rocpd_api ON rocpd_api_ops.api_id = rocpd_api.id INNER JOIN rocpd_op ON rocpd_api_ops.op_id = rocpd_op.id %s ORDER BY start/1000.0 asc"
        % (rangeStringApi, rangeStringApi, rangeStringApi)  # 应用时间范围过滤
    ):
        try:
            key = (row[2], row[3])  # Key is 'pid,tid'  # 键为'pid,tid'
            if row[0] == "0":  # Frame start  # 帧起始事件
                if key not in stacks:  # 如果该键尚无调用栈
                    stacks[key] = []  # 初始化空栈
                stack = stacks[key].append((row[1], row[4]))  # 将帧起始时间和标签入栈
                # print(f"0: new api frame: pid_tid={key} -> stack={stacks}")  # 调试打印

            elif row[0] == "1":  # Frame end  # 帧结束事件
                completed = stacks[key].pop()  # 从栈中弹出已完成的帧
                # print(f"1: end api frame: pid_tid={key} -> stack={stacks}")  # 调试打印

            elif row[0] == "2":  # API + Op  # API加操作事件
                if key in stacks and len(stacks[key]) > 0:  # 如果该键有活跃的调用栈
                    frame = stacks[key][-1]  # 获取栈顶帧
                    # print(f"2: Op on {frame} ({len(stacks[key])})")  # 调试打印
                    gpuFrame = None  # 初始化GPU帧
                    if key not in currentFrame:  # First op under the current api frame  # 当前API帧下的第一个操作
                        gpuFrame = GpuFrame()  # 创建新GPU帧
                        gpuFrame.id = frame[0]  # 设置帧ID
                        gpuFrame.name = frame[1]  # 设置帧名称
                        gpuFrame.start = row[7]  # 设置帧起始时间
                        gpuFrame.end = row[8]  # 设置帧结束时间
                        gpuFrame.gpus.append((row[5], row[6]))  # 添加GPU和队列信息
                        gpuFrame.totalOps = 1  # 操作计数初始化为1
                        # print(f"2a: new frame: {gpuFrame.gpus} {gpuFrame.start} {gpuFrame.end} {gpuFrame.end - gpuFrame.start}")  # 调试打印
                    else:
                        gpuFrame = currentFrame[key]  # 获取当前GPU帧
                        # Another op under the same frame -> union them (but only if they are butt together)  # 同一帧下的另一个操作 -> 合并（仅在时间紧密相连时）
                        if (
                            gpuFrame.id == frame[0]  # 帧ID匹配
                            and gpuFrame.name == frame[1]  # 帧名称匹配
                            and (
                                abs(row[7] - gpuFrame.end) < 200  # 新操作起始与帧结束间隔小于200
                                or abs(gpuFrame.start - row[8]) < 200  # 帧起始与新操作结束间隔小于200
                            )
                        ):
                            # if gpuFrame.id == frame[0] and gpuFrame.name == frame[1]:    # Another op under the same frame -> union them  # 同一帧下的操作 -> 合并
                            # if False:   # Turn off frame joining  # 关闭帧合并
                            if row[7] < gpuFrame.start:  # 如果新操作起始更早
                                gpuFrame.start = row[7]  # 扩展帧起始时间
                            if row[8] > gpuFrame.end:  # 如果新操作结束更晚
                                gpuFrame.end = row[8]  # 扩展帧结束时间
                            if (row[5], row[6]) not in gpuFrame.gpus:  # 如果GPU和队列未记录
                                gpuFrame.gpus.append((row[5], row[6]))  # 添加GPU和队列信息
                            gpuFrame.totalOps = gpuFrame.totalOps + 1  # 增加操作计数
                            # print(f"2c: union frame: {gpuFrame.gpus} {gpuFrame.start} {gpuFrame.end} {gpuFrame.end - gpuFrame.start}")  # 调试打印

                        else:  # This is a new frame - dump the last and make new  # 新帧 - 输出旧帧并创建新帧
                            gpuFrame = currentFrame[key]  # 获取旧帧
                            for dest in gpuFrame.gpus:  # 遍历帧涉及的每个GPU目标
                                # print(f"2: OUTPUT: dest={dest} time={gpuFrame.start} -> {gpuFrame.end} Duration={gpuFrame.end - gpuFrame.start} TotalOps={gpuFrame.totalOps}")  # 调试打印
                                outfile.write(  # 写入帧事件
                                    ',{"pid":"%s","tid":"%s","name":"%s","ts":"%s","dur":"%s","ph":"X","args":{"desc":"%s"}}\n'
                                    % (
                                        dest[0],  # GPU ID
                                        dest[1],  # 队列ID
                                        gpuFrame.name.replace('"', ""),  # 帧名称（去除引号）
                                        gpuFrame.start - 1,  # 时间戳（偏移1以区分）
                                        gpuFrame.end - gpuFrame.start + 1,  # 持续时间
                                        f"UserMarker frame: {gpuFrame.totalOps} ops",  # 描述信息
                                    )
                                )
                            currentFrame.pop(key)  # 移除旧帧

                            # make the first op under the new frame  # 为新帧创建第一个操作
                            gpuFrame = GpuFrame()  # 创建新GPU帧
                            gpuFrame.id = frame[0]  # 设置帧ID
                            gpuFrame.name = frame[1]  # 设置帧名称
                            gpuFrame.start = row[7]  # 设置帧起始时间
                            gpuFrame.end = row[8]  # 设置帧结束时间
                            gpuFrame.gpus.append((row[5], row[6]))  # 添加GPU和队列信息
                            gpuFrame.totalOps = 1  # 操作计数初始化为1
                            # print(f"2b: new frame: {gpuFrame.gpus} {gpuFrame.start} {gpuFrame.end} {gpuFrame.end - gpuFrame.start}")  # 调试打印

                    currentFrame[key] = gpuFrame  # 更新当前帧

        except ValueError:
            outfile.write("")  # 忽略值错误

    outfile.write("]\n")  # 写入事件数组结束

    if format == "object":  # 如果输出格式为对象
        outfile.write("} \n")  # 写入对象结束

    outfile.close()  # 关闭输出文件
    connection.close()  # 关闭数据库连接
