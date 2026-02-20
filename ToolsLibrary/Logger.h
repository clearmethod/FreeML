#pragma once

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace Tools 
{

class Logger 
{
public:
    enum class Level { Debug= 0, Info = 1, Warning = 2, Error = 3 };

    // RAII stream proxy that flushes on destruction
    class Stream 
    {
    public:
        explicit Stream(Level lvl) : level_(lvl) {}
        ~Stream() 
        { 
            Logger::Flush(level_, oss_.str()); 
        }

        template <typename T>
        Stream& operator<<(const T& value) 
        {
            oss_ << value;
            return *this;
        }
        // Support iostream manipulators like std::endl
        using Manip = std::ostream& (*)(std::ostream&);
        Stream& operator<<(Manip m) 
        {
            m(oss_);
            return *this;
        }

    private:
        Level level_;
        std::ostringstream oss_;
    };

    // Stream-style entry points
    static Stream Debug()   { return Stream(Level::Debug);  }
    static Stream Info()    { return Stream(Level::Info);   }
    static Stream Warning() { return Stream(Level::Warning);}
    static Stream Error()   { return Stream(Level::Error);  }

    // printf-style entry points
    static void Debugf(const char* fmt, ...)
    {
        va_list args; va_start(args, fmt); VLogf(Level::Debug, fmt, args); va_end(args);
    }

    static void Infof(const char* fmt, ...) 
    {
        va_list args; va_start(args, fmt); VLogf(Level::Info, fmt, args); va_end(args);
    }

    static void Warningf(const char* fmt, ...) 
    {
        va_list args; va_start(args, fmt); VLogf(Level::Warning, fmt, args); va_end(args);
    }

    static void Errorf(const char* fmt, ...) 
    {
        va_list args; va_start(args, fmt); VLogf(Level::Error, fmt, args); va_end(args);
    }

    // Global settings
    static void EnableConsole(bool enabled) 
    { 
        GetConsoleEnabled() = enabled; 
    }

    static void SetMinLevel(Level lvl) 
    { 
        GetMinLevel() = static_cast<int>(lvl); 
    }

    static void SetLogFile(const std::string& path) 
    {
        std::lock_guard<std::mutex> lock(GetMutex());
        auto& ofs = GetFileStream();
        ofs.close();
        if (!path.empty()) 
        {
            ofs.open(path, std::ios::out | std::ios::app);
            GetFileEnabled() = ofs.is_open();
            GetFilePath() = path;
        } 
        else
        {
            GetFileEnabled() = false;
            GetFilePath().clear();
        }
    }
    static bool IsConsoleEnabled() 
    { 
        return GetConsoleEnabled(); 
    }
    static bool IsFileEnabled() 
    { 
        return GetFileEnabled(); 
    }
    static std::string LogFilePath() 
    { 
        return GetFilePath(); 
    }

    static std::string Colour(uint32_t _r, uint32_t _g, uint32_t _b)
    {
        std::stringstream ss;
        ss << "\033[38;2;" << _r << ";"<<_g<<";"<<_b<<"m";
        return ss.str();
    }

    static std::string ClearColour()
    {
        std::stringstream ss;
        ss << "\033[0m";
        return ss.str();
    }

private:
    static void VLogf(Level lvl, const char* fmt, va_list args) 
    {
        if (!ShouldLog(lvl)) return;
        // Format using vsnprintf into a dynamic buffer
        va_list args_copy;
#if defined(_MSC_VER)
        args_copy = args;
#else
        va_copy(args_copy, args);
#endif
        int size = std::vsnprintf(nullptr, 0, fmt, args_copy);
#if !defined(_MSC_VER)
        va_end(args_copy);
#endif
        if (size <= 0) { Flush(lvl, ""); return; }
        std::vector<char> buffer(static_cast<size_t>(size) + 1);
        std::vsnprintf(buffer.data(), buffer.size(), fmt, args);
        Flush(lvl, std::string_view(buffer.data(), static_cast<size_t>(size)));
    }

    static bool ShouldLog(Level lvl) 
    {
        return static_cast<int>(lvl) >= GetMinLevel();
    }

    static void Flush(Level lvl, std::string_view text) 
    {
        if (!ShouldLog(lvl)) return;

        std::lock_guard<std::mutex> lock(GetMutex());

        const char* prefix = Prefix(lvl);
        if (GetConsoleEnabled()) 
        {
            std::ostream& os = (lvl == Level::Info) ? static_cast<std::ostream&>(std::cout)
                                                    : static_cast<std::ostream&>(std::cerr);
            os << prefix << text << '\n';
        }

        if (GetFileEnabled()) 
        {
            auto& ofs = GetFileStream();
            if (ofs.is_open()) 
            {
                ofs << prefix << text << '\n';
                ofs.flush();
            }
        }
    }

    static const char* Prefix(Level lvl) 
    {
        switch (lvl) 
        {
            case Level::Debug: return "[DEBUG] ";
            case Level::Info: return "[INFO] ";
            case Level::Warning: return "[WARN] ";
            case Level::Error: return "[ERROR] ";
        }
        return "";
    }

    // Globals (lazy-initialized, header-only safe)
    static std::mutex& GetMutex() 
    {
        static std::mutex m;
        return m;
    }

    static int& GetMinLevel() 
    {
        static int lvl = static_cast<int>(Level::Info);
        return lvl;
    }
    static bool& GetConsoleEnabled() 
    {
        static bool enabled = true; // default to console
        return enabled;
    }

    static bool& GetFileEnabled() 
    {
        static bool enabled = false;
        return enabled;
    }
    
    static std::string& GetFilePath() 
    {
        static std::string path;
        return path;
    }

    static std::ofstream& GetFileStream() 
    {
        static std::ofstream ofs;
        return ofs;
    }
};

} // namespace Tools

// Convenience macros for stream-style logging
#define LOG_DEBUG()   ::Tools::Logger::Debug()
#define LOG_INFO()    ::Tools::Logger::Info()
#define LOG_WARNING() ::Tools::Logger::Warning()
#define LOG_ERROR()   ::Tools::Logger::Error()

// Convenience macros for printf-style logging
#define LOG_DEBUGF(...)   ::Tools::Logger::Debugf(__VA_ARGS__)
#define LOG_INFOF(...)    ::Tools::Logger::Infof(__VA_ARGS__)
#define LOG_WARNINGF(...) ::Tools::Logger::Warningf(__VA_ARGS__)
#define LOG_ERRORF(...)   ::Tools::Logger::Errorf(__VA_ARGS__)
