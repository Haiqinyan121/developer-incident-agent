# Go goroutine 泄漏排障手册

goroutine 数量持续增长且请求结束后不回落，通常说明任务在 channel、锁、网络读取或未取消的 context 上阻塞。

排查时对比不同时刻的 goroutine profile，寻找重复堆栈；检查 channel 是否存在无人接收或无人发送的路径；确认超时和取消信号能够传递到后台任务。修复后应通过重复压测确认 goroutine 数量能够回落到稳定区间。
