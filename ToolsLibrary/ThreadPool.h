#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <latch>

#include "Logger.h"

class ThreadPool
{
public:
    explicit ThreadPool(size_t threadCount = std::thread::hardware_concurrency())
    {
        Start(threadCount == 0 ? 1 : threadCount);
        LOG_INFO() << "Initialising threadpool with " << threadCount << " available threads.";
    }

    ~ThreadPool()
    {
        Stop();
    }

    template<typename F>
    void Enqueue(F&& task)
    {
        {
            std::unique_lock<std::mutex> lock(m_eventMutex);
            m_tasks.emplace(std::forward<F>(task));
        }
        m_eventVar.notify_one();
    }

    size_t GetThreadCount() const { return m_threads.size(); }

    template<typename F>
    void ParallelFor(uint32_t count, F&& fn)
    {
        if (count == 0) return;
        std::latch latch(count);
        for (uint32_t i = 0; i < count; ++i)
        {
            Enqueue([&, i]
            {
                fn(i);
                latch.count_down();
            });
        }
        latch.wait();
    }

private:
    void Start(size_t threadCount)
    {
        for(size_t i = 0; i < threadCount; ++i)
        {
            m_threads.emplace_back([this]
            {
                while(true)
                {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(m_eventMutex);
                        m_eventVar.wait(lock, [this]{ return m_stopping || !m_tasks.empty(); });
                        if(m_stopping && m_tasks.empty())
                            return;
                        task = std::move(m_tasks.front());
                        m_tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    void Stop()
    {
        {
            std::unique_lock<std::mutex> lock(m_eventMutex);
            m_stopping = true;
        }
        m_eventVar.notify_all();
        for(std::thread& t : m_threads)
        {
            if(t.joinable())
                t.join();
        }
    }

    std::vector<std::thread> m_threads;
    std::queue<std::function<void()>> m_tasks;
    std::mutex m_eventMutex;
    std::condition_variable m_eventVar;
    bool m_stopping = false;
};

inline ThreadPool* GetGlobalThreadPool()
{
    static ThreadPool pool(std::thread::hardware_concurrency());
    return &pool;
}

template<typename F>
void ParallelFor(uint32_t count, F&& fn)
{
    if (count == 0) return;
    std::latch latch(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        Enqueue([&, i]
        {
            fn(i);
            latch.count_down();
        });
    }
    latch.wait();
}


#endif // THREAD_POOL_H
