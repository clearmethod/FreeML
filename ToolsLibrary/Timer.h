#pragma once

#include <ToolsLibrary/Logger.h>
#include <chrono>

#define LOCALTIMER(name) Timer t_##name(#name, true)

class Timer 
{
public:
    Timer(std::string _name = "", bool _printOnDelete = false): m_printOnDelete(_printOnDelete), m_name(_name) { Start(); }

    ~Timer()
    {
        if(m_printOnDelete)
        {
            PrintElapsed(m_name);
        }
    }
    void Start() 
    {
        m_startTime = std::chrono::high_resolution_clock::now();
    }

    // Returns elapsed time in seconds (as float)
    double Elapsed() const 
    {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = now - m_startTime;
        return duration.count();
    }

    // Prints the elapsed time
    void PrintElapsed(const std::string& label = "Elapsed") const 
    {
        LOG_INFO() << label << ": \t" << Elapsed() << " seconds";
    }

private:
    std::chrono::high_resolution_clock::time_point m_startTime;
    bool        m_printOnDelete     = false;
    std::string m_name              = "";
};